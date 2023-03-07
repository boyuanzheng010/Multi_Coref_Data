import json

import jsonlines
import spacy, benepar
import pickle as pkl
from tqdm import tqdm
from supar.utils import Tree
from copy import deepcopy
from annotation.multilingual_coref_agreement.visulization_util import extract_scene_info, add_speaker_to_utt, \
    extract_tokens_speakers

with open('data/parallel_corpus/split_dict.pkl', 'rb') as f:
    all_scene_ids = pkl.load(f)['test']
print(all_scene_ids)
print(len(all_scene_ids))

# nlp = spacy.load("zh_core_web_sm")
# nlp.add_pipe("benepar", config={"model": "benepar_zh2"})

# with open('data/parallel_corpus/all_coref_data_en.json', 'r') as f:
#     data = json.load(f)

parser = benepar.Parser("benepar_zh2")

data = []
with open('data/exp_inputs/test.chinese.512.jsonlines', 'r') as f:
    reader = jsonlines.Reader(f)
    for line in reader:
        scene_id = line['doc_key'][:10]
        tokens = line['tokens']
        # _, speaker_list = extract_tokens_speakers(line)

        # Collect Token-Level Utterances
        utt_idxs = {}
        speaker_idxs = {}
        all_speakers = []
        for item in line['speakers']:
            all_speakers.extend(item)
        sentence_map = line['sentence_map']
        subtoken_map = line['subtoken_map']
        for idx in range(len(subtoken_map)):
            sent_idx = sentence_map[idx]
            token_idx = subtoken_map[idx]
            if sent_idx not in utt_idxs:
                utt_idxs[sent_idx] = [token_idx]
            else:
                utt_idxs[sent_idx].append(token_idx)

            # if (sent_idx not in speaker_idxs) and (all_speakers[idx]!='[SPL]'):
            if sent_idx not in speaker_idxs:
                speaker_idxs[sent_idx] = set()
                speaker_idxs[sent_idx].add(all_speakers[idx])
            else:
                speaker_idxs[sent_idx].add(all_speakers[idx])

        for sent_idx in utt_idxs:
            utt_idxs[sent_idx] = sorted(list(set(utt_idxs[sent_idx])))

        utterances = []
        for sent_idx in utt_idxs:
            temp_sent = []
            for idx in utt_idxs[sent_idx]:
                temp_sent.append(tokens[idx])
            utterances.append(temp_sent)

        speaker_list = []
        for sent_idx in speaker_idxs:
            current_speaker = list(speaker_idxs[sent_idx])
            if '[SPL]' in current_speaker:
                current_speaker.remove('[SPL]')
            if len(current_speaker) != 0:
                speaker_list.append(current_speaker[0])

        data.append({
            "zh_subtitles": utterances,
            "scene_id": scene_id,
            "speakers": speaker_list
        })

for i, line in tqdm(enumerate(data)):
    try:
        scene_id = line['scene_id']
        if scene_id not in all_scene_ids:
            continue
        # if scene_id != "s09e07c07t":
        #     continue
        print(scene_id)
        zh_utts = line['zh_subtitles']
        speaker_list = line['speakers']

        collected_scene = []
        for speaker, utt in zip(speaker_list, zh_utts):
            # temp = data[line['scene_id']]
            # print(speaker, utt)
            input_sentence = benepar.InputSentence(utt)
            tree = parser.parse(input_sentence)
            factorized_tree = Tree.factorize(tree)
            constituents = []
            for start, end, tag in factorized_tree:
                constituents.append((" ".join(utt[start: end]), start, end, tag))

            prons = deepcopy(constituents)
            collected_scene.append({
                "pron": prons,
                "constituent": constituents,
                "speaker": speaker,
                "utterance": utt
            })
        with open('data/parsed_data/zh/'+scene_id+".pkl", 'wb') as f:
            pkl.dump(collected_scene, f)
    except:
        print("Pass", line['scene_id'][:10])
        continue

# with open('data/parsed_data/zh_all.pkl', 'wb') as f:
#     pkl.dump(all_data, f)
