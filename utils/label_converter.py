import torch
import pickle
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BeamEntry:
    def __init__(self):
        self.prTotal = 0
        self.prNonBlank = 0
        self.prBlank = 0
        self.prText = 1
        self.lmApplied = False
        self.labeling = ()


class BeamState:
    def __init__(self):
        self.entries = {}

    def norm(self):
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]

    def wordsearch(self, classes, ignore_idx, beamWidth, dict_list):
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)[:beamWidth]

        for j, candidate in enumerate(sortedBeams):
            idx_list = candidate.labeling
            text = ''
            for i,l in enumerate(idx_list):
                if l not in ignore_idx and (not (i > 0 and idx_list[i - 1] == idx_list[i])):
                    text += classes[l]

            if j == 0: best_text = text
            if text in dict_list:
                print('found text: ', text)
                best_text = text
                break
            else:
                print('not in dict: ', text)
        return best_text


def addBeam(beamState, labeling):
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, ignore_idx, lm, beamWidth=25, dict_list = []):
    blankIdx = 0
    maxT, maxC = mat.shape

    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    for t in range(maxT):

        curr = BeamState()
        bestLabelings = last.sort()[0:beamWidth]

        for labeling in bestLabelings:
            prNonBlank = 0

            if labeling:
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]
            addBeam(curr, labeling)
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText
            curr.entries[labeling].lmApplied = True

            for c in range(maxC - 1):
                newLabeling = labeling + (c,)

                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                addBeam(curr, newLabeling)

                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

        last = curr

    last.norm()

    if dict_list == []:
        bestLabeling = last.sort()[0]
        res = ''
        for i,l in enumerate(bestLabeling):
            if l not in ignore_idx and (not (i > 0 and bestLabeling[i - 1] == bestLabeling[i])):
                res += classes[l]
    else:
        res = last.wordsearch(classes, ignore_idx, beamWidth, dict_list)

    return res



def consecutive(data, mode ='first', stepsize=1):
    group = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    group = [item for item in group if len(item)>0]

    if mode == 'first': result = [l[0] for l in group]
    elif mode == 'last': result = [l[-1] for l in group]
    return result


def word_segmentation(mat, separator_idx =  {'th': [1,2],'en': [3,4]}, separator_idx_list = [1,2,3,4]):
    result = []
    sep_list = []
    start_idx = 0
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0: mode ='first'
        else: mode ='last'
        a = consecutive( np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [ [item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]:
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]:
                if sep_lang == lang:
                    new_sep_pair = [lang, [sep_start_idx+1, sep[0]-1]]
                    if sep_start_idx > start_idx:
                        result.append( ['', [start_idx, sep_start_idx-1] ] )
                    start_idx = sep[0]+1
                    result.append(new_sep_pair)
                else: # reset
                    sep_lang = ''

    if start_idx <= len(mat)-1:
        result.append( ['', [start_idx, len(mat)-1] ] )
    return result


class CTCLabelConverter(object):
    def __init__(self, character, separator_list = {}, dict_pathlist = {}):
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character
        self.separator_list = separator_list

        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep

        self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

        dict_list = {}
        for lang, dict_path in dict_pathlist.items():
            with open(dict_path, "rb") as input_file:
                word_count = pickle.load(input_file)
            dict_list[lang] = word_count
        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return torch.IntTensor(text), torch.IntTensor(length)

    def decode_greedy(self, text_index, length):
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] not in self.ignore_idx and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

    def decode_beamsearch(self, mat, beamWidth=5):
        texts = []

        for i in range(mat.shape[0]):
            t = ctcBeamSearch(mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth)
            texts.append(t)
        return texts

    def decode_wordbeamsearch(self, mat, beamWidth=5):
        texts = []
        argmax = np.argmax(mat, axis = 2)
        for i in range(mat.shape[0]):
            words = word_segmentation(argmax[i])
            string = ''
            for word in words:
                matrix = mat[i, word[1][0]:word[1][1]+1,:]
                if word[0] == '': dict_list = []
                else: dict_list = self.dict_list[word[0]]
                t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None, beamWidth=beamWidth, dict_list=dict_list)
                string += t
            texts.append(string)
        return texts



class Averager(object):

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res