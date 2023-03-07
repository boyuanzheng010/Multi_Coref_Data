import json
import spacy, benepar
import pickle as pkl
from tqdm import tqdm

with open('data/parallel_corpus/split_dict.pkl', 'rb') as f:
    all_scene_ids = pkl.load(f)['test']
print(all_scene_ids)
print(len(all_scene_ids))

berkeley_parser = spacy.load('en_core_web_md')
berkeley_parser.add_pipe('benepar', config={'model': 'benepar_en3_large'})

with open('data/parallel_corpus/all_coref_data_en.json', 'r') as f:
    data = json.load(f)

all_data = {}
for i, line in tqdm(enumerate(data)):
    scene_id = line['scene_id'][:10]
    if scene_id not in all_scene_ids:
        continue
    # if scene_id != "s09e07c07t":
    #     continue

    # Collect Speakers
    speaker_list = []
    en_utts = []
    for utt in line['sentences']:
        speaker_list.append(" ".join(utt[: utt.index(":")]))
        en_utts.append(" ".join(utt[utt.index(":")+1:]))
    print(speaker_list)
    print(en_utts)

    collected_scene = []
    for speaker, utt in zip(speaker_list, en_utts):
        if utt == " ":
            utt = ""
        doc = berkeley_parser(utt)
        constituents = []
        for sent in list(doc.sents):
            for token in sent._.constituents:
                constituents.append((token.text, token.start, token.end, token._.labels))
        prons = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in enumerate(doc)]
        collected_scene.append({
            "pron": prons,
            "constituent": constituents,
            "speaker": speaker,
            "utterance": utt
        })
    all_data[scene_id] = collected_scene

with open('data/parsed_data/en_all.pkl', 'wb') as f:
    pkl.dump(all_data, f)
