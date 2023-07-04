from utils.data import prepareData, tensorsFromPair, tensorFromSentence, MAX_LENGTH, EOS_token, SOS_token
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils.plots import showPlot
from utils.time import timeSince, asMinutes

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
device = "cpu"


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, 10)
        self.gru = nn.GRU(10, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


# train on individual sentence
def train(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, 256)

    loss = 0

    for i in range(input_length):
        output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = output[0, 0]

    decoder_input = torch.Tensor([[SOS_token]]).long()
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < 0.5 else False

    if use_teacher_forcing:

        # next input would be from target tensor
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]
    else:
        # use our own predictions as next input
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[i])

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == EOS_token: break

    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_length


enc = EncoderRNN(input_lang.n_words, 256)
dec = DecoderRNN(256, output_lang.n_words)


def start_training():
    enc_optim = torch.optim.SGD(enc.parameters(), lr=0.01)
    dec_optim = torch.optim.SGD(dec.parameters(), lr=0.01)

    criterion = nn.NLLLoss()

    n_iters = 75000

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every = 5000
    plot_every = 5000
    plot_losses = []

    start = time.time()

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(n_iters)]
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, enc, dec, enc_optim, dec_optim, criterion)
        print(loss)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            torch.save(enc, "models/encoder.pt")
            torch.save(dec, "models/decoder.pt")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    enc = torch.load("models/encoder.pt")
    dec = torch.load("models/decoder.pt")
    evaluateRandomly(enc, dec)
