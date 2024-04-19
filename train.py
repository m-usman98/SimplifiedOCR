import argparse
import numpy as np
import torch
from network.model_arch import Model
import torch.optim as optim
from dataloader import load_dataset
from utils.label_converter import CTCLabelConverter, Averager
from utils.utils import model_parameters_cal
from network.init_weights import init_modeL_weights
from validation import validation

parser = argparse.ArgumentParser(description="Mobile OCR Project")
parser.add_argument("--NUM_WORKERS", default=1, type=int)
parser.add_argument("--IMAGE_HEIGHT", default=64, type=int)
parser.add_argument("--IMAGE_WIDTH", default=600, type=int)
parser.add_argument("--EPOCHS", default=300, type=int)
parser.add_argument("--BATCH_SIZE", default=32, type=int)
parser.add_argument("--MAX_BATCH_SIZE", default=34, type=int)
parser.add_argument("--LR", default=1., type=float)
parser.add_argument("--INPUT_CHANNELS", default=3, type=int)
parser.add_argument("--OUTPUT_CHANNELS", default=256, type=int)
parser.add_argument("--HIDDEN_SIZE", default=256, type=int)
parser.add_argument("--TRAIN_DATA_DIR", default="./dataset/Train", type=str)
parser.add_argument("--VAL_DATA_DIR", default="./dataset/Val", type=str)
parser.add_argument("--SAVE_DIR", default="./checkpoints/best_model.pth'", type=str)
parser.add_argument("--NUMBER", default="0123456789", type=str)
parser.add_argument("--SYMBOL", default="!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €", type=str)
parser.add_argument("--LANG_CHAR", default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", type=str)
parser.add_argument("--VALIDATION_ITER", default=30, type=int)

args = parser.parse_args()


def train(model, train_dataset, val_dataset, converter, device, epoch):
    optimizer = optim.Adadelta(model.parameters(), lr=args.LR, rho=0.95, eps=0.00000001)
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    loss_avg = Averager()
    accuracy = 0
    model.train()
    for batch_idx, (images, labels) in enumerate(train_dataset):
        encoded_text, encoded_length = converter.encode(labels, batch_max_length=args.BATCH_SIZE + 2)
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        dynamic_batch_size = images.size(0)
        prediction = model(images, encoded_text).log_softmax(2)
        prediction_size = torch.IntTensor([prediction.size(1)] * dynamic_batch_size)
        prediction = prediction.permute(1, 0, 2)
        torch.backends.cudnn.enabled = False
        loss = criterion(prediction, encoded_text.to(device), prediction_size.to(device), encoded_length.to(device))
        torch.backends.cudnn.enabled = True
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        loss_avg.add(loss)
    print("Training Loss: ", loss_avg.val())

    if epoch % args.VALIDATION_ITER == 0:
        valid_loss, accuracy, norm_ed_accuracy = validation(model, val_dataset, converter, device, args)
        print(f"Valid Loss: {valid_loss:.3f}, Accuracy: {accuracy:.3f}, Norm ED Accuracy: {norm_ed_accuracy:.3f} ")
        model.train()
        return norm_ed_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The Detected Device: ", torch.cuda.get_device_name(device))

    converter = CTCLabelConverter(args.NUMBER + args.SYMBOL + args.LANG_CHAR)
    num_classes = len(converter.character)

    load_train_dataset = load_dataset(args.TRAIN_DATA_DIR, True, args.IMAGE_HEIGHT, args.IMAGE_WIDTH)
    print("The Total Number Of Train Images Are: ", len(load_train_dataset))
    load_val_dataset = load_dataset(args.VAL_DATA_DIR, None, args.IMAGE_HEIGHT, args.IMAGE_WIDTH)
    print("The Total Number Of Valid Images Are:", len(load_val_dataset))

    print("The Total Number of Classes Are: ", num_classes)

    loaded_train_tr_dataset = torch.utils.data.DataLoader(load_train_dataset, batch_size=args.BATCH_SIZE, shuffle=True,
                                                          num_workers=args.NUM_WORKERS)
    load_val_tr_dataset = torch.utils.data.DataLoader(load_val_dataset, batch_size=args.BATCH_SIZE, shuffle=True,
                                                      num_workers=args.NUM_WORKERS)

    data_itr = iter(load_val_tr_dataset)
    img, label = next(data_itr)

    model = Model(args.INPUT_CHANNELS, args.OUTPUT_CHANNELS, args.HIDDEN_SIZE, num_classes).to(device)
    model = init_modeL_weights(model, device)

    model_parameters_cal(model)

    best_loss = np.inf
    for epoch in range(1, args.EPOCHS):
        print('\nTraining Epoch: %d' % epoch)
        current_loss = train(model, loaded_train_tr_dataset, load_val_tr_dataset, converter, device, epoch)
        if current_loss is not None and current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), args.SAVE_DIR)
            print("saving the model...")
