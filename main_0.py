import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm
from tricks import FGM, EMA
import torchcontrib


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):

    K_foldNum = 10  # 设置10折交叉验证
    for i in range(K_foldNum):
        # 1. load data
        train_dataloader, val_dataloader = create_dataloaders(args, i)

        # 2. build model and optimizers
        model = MultiModal(args)
        base_optimizer, scheduler = build_optimizer(args, model)
        optimizer = torchcontrib.optim.SWA(base_optimizer)
        if args.device == "cuda":
            model = torch.nn.parallel.DataParallel(model.to(args.device))

        # 3. training
        step = 0
        is_exist_model = ""  # judge model exist
        best_score = args.best_score
        start_time = time.time()
        num_total_steps = len(train_dataloader) * args.max_epochs
        fgm = FGM(model, eps=0.5)  # FGM
        ema = EMA(model, 0.9995)  # EMA
        ema.register()
        valid_f1 = 0
        f = open("training log.txt", "w")
        for epoch in range(args.max_epochs):
            loop = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in loop:
                model.train()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

                fgm.attack()
                loss_adv, _, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()

                optimizer.step()
                ema.update()

                optimizer.zero_grad()
                scheduler.step()

                loop.set_description(f"Epoch [{epoch}]")
                loop.set_postfix(
                    loss=loss.item(), acc=accuracy.item(), valid_f1=valid_f1
                )

                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime(
                        "%H:%M:%S", time.gmtime(remaining_time)
                    )
                    f.write(
                        f"Fold {i} Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}, lr {optimizer.param_groups[0]['lr']:.3e}"
                        + "\n"
                    )

                # 4-1. validation
                if step % 500 == 0:
                    ema.apply_shadow()  # eam
                    loss, results = validate(model, val_dataloader)
                    results = {k: round(v, 4) for k, v in results.items()}
                    valid_f1 = results["mean_f1"]
                    f.write(
                        f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}" + "\n"
                    )
                    # 5-1. save checkpoint
                    mean_f1 = results["mean_f1"]
                    if mean_f1 > best_score:
                        best_score = mean_f1
                        
                        if is_exist_model != "":
                            os.remove(is_exist_model)
                        torch.save(
                            {
                                "Fold": i,
                                "epoch": epoch,
                                "model_state_dict": model.module.state_dict(),
                                "mean_f1": mean_f1,
                            },
                            f"{args.savedmodel_path}/model_Fold_{i}_epoch_{epoch}_mean_f1_{mean_f1}.bin",
                        )
                        is_exist_model = args.savedmodel_path + 'model_Fold_' + \
                            str(i) + '_epoch_' + str(epoch) + \
                            '_mean_f1_' + str(mean_f1) + '.bin'
                    ema.restore()  # ema

                if epoch > 1 and step % 500 == 0:
                    ema.apply_shadow()
                    optimizer.update_swa()
                    ema.restore()

        print("swa_evaluate")
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_dataloader, model)
        # 4-2. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        valid_f1 = results["mean_f1"]
        f.write(
            f"Fold {i} Epoch {epoch} step {step}: loss {loss:.3f}, {results}" + "\n"
        )
        print(results)

        # 5-2. save checkpoint
        mean_f1 = results["mean_f1"]
        if mean_f1 > best_score:
            best_score = mean_f1
            if is_exist_model != "":
                os.remove(is_exist_model)
            torch.save(
                {
                    "Fold": i,
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "mean_f1": mean_f1,
                },
                f"{args.savedmodel_path}/model_Fold_{i}_epoch_{epoch}_mean_f1_{mean_f1}.bin",
            )
            is_exist_model = args.savedmodel_path + 'model_Fold_' + \
                str(i) + '_epoch_' + str(epoch) + \
                '_mean_f1_' + str(mean_f1) + '.bin'


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == "__main__":
    main()
