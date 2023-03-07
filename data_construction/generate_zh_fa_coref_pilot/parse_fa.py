import json
import spacy, benepar
import pickle as pkl
from tqdm import tqdm
import dadmatools.pipeline.language as language

with open('data/parallel_corpus/split_dict.pkl', 'rb') as f:
    all_scene_ids = pkl.load(f)['test']
print(all_scene_ids)
print(len(all_scene_ids))

pips = 'tok,lem,pos,dep,cons'
nlp = language.Pipeline(pips)

with open('data/parallel_corpus/all_coref_data_en.json', 'r') as f:
    data = json.load(f)

all_data = {}
for i, line in tqdm(enumerate(data)):
    scene_id = line['scene_id'][:10]
    if scene_id not in all_scene_ids:
        continue
    # if scene_id != "s09e07c07t":
    #     continue
    print(scene_id)
    en_utts = line['sentences']
    fa_utts = line['fa_subtitles']
    # Collect Speakers
    speaker_list = []
    for utt in en_utts:
        speaker_list.append(" ".join(utt[: utt.index(":")]))

    collected_scene = []
    for speaker, utt in zip(speaker_list, fa_utts):
        if utt == " ":
            utt = ""
        doc = nlp(utt)
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

with open('data/parsed_data/fa_all.pkl', 'wb') as f:
    pkl.dump(all_data, f)
