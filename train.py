import os
import sys
import json
import boto3
import random
import logging
import argparse
import torch

import matplotlib.pyplot as plt

from model import CaptioningModel
from data import get_data_loader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sagemaker.session import Session
from sagemaker.experiments.run import load_run

CAP_LENGTH = 16

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(args):
    vocab = json.load(open(os.path.join(args.text, 'vocab.json'), 'r', encoding='utf-8'))
    
    device = torch.device("cuda")
    
    train_loader = get_data_loader(args.train, args.text, 'train2017', args.batch_size)
    val_loader = get_data_loader(args.test, args.text, 'val2017', args.batch_size)
    train_val_loader = get_data_loader(args.train, args.text, 'train2017', args.batch_size, length=len(val_loader.sampler))
    
    model = CaptioningModel(vocab_size=len(vocab),
                        d_model=512,
                        nheads=args.heads,
                        nlayers=args.nlayers)

    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    session = Session(boto3.session.Session(region_name=args.region))
    with load_run(sagemaker_session=session) as run:
    
        run.log_parameters(
            {
                "batch_size": args.batch_size,
                "lr": args.lr,
                "heads": args.heads,
                "nlayers": args.nlayers
            }
        )
    
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            for i, (source, target) in enumerate(train_loader):
                source, target = source.to(device), target.to(device)

                target_in = target[:, :-1]
                target_out = target[:, 1:]
                mask = target_out != 0

                optimizer.zero_grad()

                scores = model(source, target_in)
                if i == 0 and epoch == 1:
                    print(scores.get_device())
                loss = transformer_temporal_softmax_loss(scores, target_out, mask)

                loss.backward()
                optimizer.step()

                run.log_metric(name='train loss', value=loss.item(), step=i)

                if i % args.log_interval == 0:
                    logger.info(
                        "epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}".format(
                            epoch,
                            i * source.shape[0],
                            len(train_loader.sampler),
                            100.0 * i / len(train_loader),
                            loss.item(),
                        )
                    )

            validate(model, val_loader, vocab, run, 'test', epoch, args.log_interval)
            validate(model, train_val_loader, vocab, run, 'train', epoch, args.log_interval)

        model_path = os.path.join(args.model, 'checkpoint.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)
        run.log_artifact(name='checkpoint', value=model_path)


def validate(model, data_loader, vocab, run, split, epoch, log_interval):
    logger.info("validating...")
    model.eval()
    device = torch.device("cuda")
    
    smoothing = SmoothingFunction()
    meta_tokens = [0, 1, 2]
    
    total_score = 0
    batches = 0
    with torch.no_grad():
        for batch_idx, (source, target) in enumerate(data_loader):
            source, target = source.to(device), target.to(device)
            target = target[:, 1:]

            batch_size = target.shape[0]
            captions = torch.zeros((batch_size, CAP_LENGTH), dtype=torch.int32, device=device)
            _target = torch.ones((batch_size, 1), dtype=torch.int32, device=device)

            for i in range(CAP_LENGTH):
                scores = model.forward(source, _target)
                scores = scores[:, -1, :]

                predictions = torch.argmax(scores, axis=1)
                captions[:, i] = predictions

                predictions = predictions.unsqueeze(1)
                _target = torch.cat([_target, predictions], dim=1)
    
            idxs = random.sample(range(batch_size), 5) if batch_size >= 5 else [] # randomly sample 5 captions to view
            fig, axs = plt.subplots(1, 5, figsize=(25, 5))
            subplot_idx = 0
            for i in range(batch_size):
                reference = [vocab[str(token)] for token in target[i, :].tolist() if token not in meta_tokens]
                hypothesis = [vocab[str(token)] for token in captions[i, :].tolist() if token not in meta_tokens]
                bleu_score = round(sentence_bleu([reference], hypothesis, smoothing_function=smoothing.method3), ndigits=4)
                
                if i in idxs and batch_idx % log_interval == 0:
                    image = source[i, ...].cpu().permute(1, 2, 0)
                    axs[subplot_idx].imshow(image)
                    
                    axs[subplot_idx].text(0.5, 1.05, ' '.join(reference), fontsize=8, ha='center', transform=axs[subplot_idx].transAxes)
                    axs[subplot_idx].text(0.5, -.05, f'{bleu_score}: ' + ' '.join(hypothesis), fontsize=8, ha='center', transform=axs[subplot_idx].transAxes)
                    axs[subplot_idx].axis('off')
                    
                    subplot_idx += 1
                    
                total_score += bleu_score
                batches += 1
            
            if batch_idx % log_interval == 0:
                filename = f'{split}_{epoch}_{batch_idx}'
                path = f'/opt/ml/output/data/{filename}.png'
                plt.savefig(path)
                run.log_file(path, filename)
                
            plt.close()
                
    avg_score = round(total_score / batches, ndigits=4)
    run.log_metric(name=f'{split} accuracy', value=avg_score, step=epoch)
    logger.info(f'{split} set bleu score: {avg_score}')
            
            
# From https://cs231n.github.io/assignments2023/assignment3/
def transformer_temporal_softmax_loss(x, y, mask):
        """
        Inputs:
        - x: Input scores, of shape (N, T, V)
        - y: Ground-truth indices, of shape (N, T) where each element is in the range
             0 <= y[i, t] < V
        - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
          the scores at x[i, t] should contribute to the loss.

        Returns a tuple of:
        - loss: Scalar giving loss
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss            

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, metavar="LR", help="learning rate"
    )
    
    parser.add_argument(
        "--heads", type=int, default=2, metavar="N", help="number of heads for multi-head attention"
    )
    parser.add_argument(
        "--nlayers", type=int, default=2, metavar="N", help="number of transformer decoder layers"
    )
    
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="path to model checkpoint"
    )

    parser.add_argument("--region", type=str, default="us-east-1")
    
    parser.add_argument("--model", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--text", type=str, default=os.environ["SM_CHANNEL_TEXT"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())