import torch
import torch.nn.functional as F
from network.model_arch import Model
from dataloader import load_dataset
from utils.label_converter import CTCLabelConverter, Averager
import argparse
from nltk.metrics.distance import edit_distance
from network.init_weights import init_modeL_weights

parser = argparse.ArgumentParser(description="Mobile OCR Project")
parser.add_argument("--NUM_WORKERS", default=1, type=int)
parser.add_argument("--IMAGE_HEIGHT", default=64, type=int)
parser.add_argument("--IMAGE_WIDTH", default=600, type=int)
parser.add_argument("--BATCH_SIZE", default=32, type=int)
parser.add_argument("--MAX_BATCH_SIZE", default=34, type=int)
parser.add_argument("--INPUT_CHANNELS", default=3, type=int)
parser.add_argument("--OUTPUT_CHANNELS", default=256, type=int)
parser.add_argument("--HIDDEN_SIZE", default=256, type=int)
parser.add_argument("--VAL_DATA_DIR", default="./dataset/Val", type=str)
parser.add_argument("--NUMBER", default="0123456789", type=str)
parser.add_argument("--SYMBOL", default="!\"#$%&'()*+,-./€[]{}", type=str)
parser.add_argument("--LANG_CHAR", default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", type=str)
parser.add_argument("--SAVED_MODEL", default="./checkpoints/best_model.pth", type=str)


args = parser.parse_args()


def test(model, test_dataset, converter, device):
    correct = 0
    norm_ED = 0
    length_of_data = 0
    model.eval()
    for batch_idx, (images, labels) in enumerate(test_dataset):
        # plt.imshow(images[0].permute(1, 2, 0))
        text_for_prediction = torch.LongTensor(args.BATCH_SIZE, args.MAX_BATCH_SIZE, + 1).fill_(0).to(device)
        images = images.to(device)
        dynamic_batch_size = images.size(0)
        length_of_data = length_of_data + dynamic_batch_size

        predictions = model(images, text_for_prediction)
        prediction_size = torch.IntTensor([predictions.size(1)] * dynamic_batch_size).to(device)

        _, predictions_index = predictions.max(2)
        predictions_index = predictions_index.view(-1)
        predictions_str = converter.decode_greedy(predictions_index.data, prediction_size.data)
        print(f"GT: {labels[:3]},       Prediction: {predictions_str[:3]}")

        predictions_prob = F.softmax(predictions, dim=2)
        predictions_max_prob, _ = predictions_prob.max(dim=2)
        confidence_score_list = []

        for gt, pred, pred_max_prob in zip(labels, predictions_str, predictions_max_prob):
            if pred == gt:
                correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
            try:
                confidence_score = predictions_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

    accuracy = correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)
    print("==============================================")
    print(f"Accuracy: {accuracy:.3f}, Norm ED Accuracy: {norm_ED:.3f} ")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The Detected Device: ", torch.cuda.get_device_name(device))

    converter = CTCLabelConverter(args.NUMBER + args.SYMBOL + args.LANG_CHAR)
    num_classes = len(converter.character)

    load_val_dataset = load_dataset(args.VAL_DATA_DIR, None, args.IMAGE_HEIGHT, args.IMAGE_WIDTH)
    print("The Total Number Of Valid Images Are:", len(load_val_dataset))
    print("The Total Number of Classes Are: ", num_classes)

    loaded_valid_tr_dataset = torch.utils.data.DataLoader(load_val_dataset, batch_size=args.BATCH_SIZE, shuffle=True,
                                                          num_workers=args.NUM_WORKERS)

    model = Model(args.INPUT_CHANNELS, args.OUTPUT_CHANNELS, args.HIDDEN_SIZE, num_classes).to(device)
    model = init_modeL_weights(model, device)

    model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=device))

    test(model, loaded_valid_tr_dataset, converter, device)
