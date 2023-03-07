import json
import spacy, benepar
import pickle as pkl
from tqdm import tqdm
import dadmatools.pipeline.language as language
import jsonlines
import nltk
from supar.utils import Tree

pips = 'tok,lem,pos,dep,cons'
nlp = language.Pipeline(pips)

data = []
with open('data/exp_inputs/test.farsi.512.jsonlines') as f:
    for i, instance in tqdm(enumerate(jsonlines.Reader(f))):
        data.append(instance)

for i, line in tqdm(enumerate(data)):
    scene_id = line['doc_key'].strip().split('_')[0]
    tokens = line['tokens']
    subtoken_map = line['subtoken_map']
    sentence_map = line['sentence_map']
    # if scene_id not in ['s09e03c09t', 's09e03c10t', 's09e03c11t', 's09e03c12t', 's09e03c13t', 's09e04c00t', 's09e04c01t', 's09e04c02t', 's09e04c03t', 's09e04c04t', 's09e04c05t', 's09e04c06t', 's09e04c07t', 's09e04c08t', 's09e04c09t']:
    #     continue
    # if scene_id !="s09e04c09t":
    #     continue
    print(scene_id)

    fa_utts = []
    for sent_id in sorted(list(set(sentence_map))):
        # For each utterance sentence
        # Locate token_idxs, subtoken_idxs in model input
        token_idxs = []
        subtoken_idxs = []
        for subtoken_idx in range(len(sentence_map)):
            if sentence_map[subtoken_idx]==sent_id:
                token_idxs.append(subtoken_map[subtoken_idx])
                subtoken_idxs.append(subtoken_idx)

        # Align constituents indexs to model input indexs
        # Retrieve token and subtoken texts
        utt_tokens = []
        for token_idx in list(set(token_idxs)):
            utt_tokens.append(tokens[token_idx])
        fa_utts.append(" ".join(utt_tokens))

    # # Collect Utterances
    # utt_idxs = list(set(data[0]['sentence_map']))
    # fa_utts = []
    # for utt_idx in utt_idxs:
    #     # Locate all subtoken_idxs belonging to the same sentence
    #     utterance_subtoken_idxs = []
    #     for subtoken_idx in range(len(subtoken_map)):
    #         if sentence_map[subtoken_idx]==utt_idx:
    #             utterance_subtoken_idxs.append(subtoken_idx)
    #     # Retrieve all token indexs belonging to the same sentences
    #     utterance_token_idxs = []
    #     for subtoken_idx in utterance_subtoken_idxs:
    #         utterance_token_idxs.append(subtoken_map[subtoken_idx])
    #     # Collect tokens and join into a string for constituent parsing
    #     utterance_tokens = []
    #     for token_idx in sorted(list(set(utterance_token_idxs))):
    #         utterance_tokens.append(tokens[token_idx])
    #     fa_utts.append(" ".join(utterance_tokens))

    # Perform Parsing and Collect Results
    collected_scene = []
    for utt in fa_utts:
        try:
            doc = nlp(utt)
            constituents = []
            for sent_constituency in doc._.constituency:
                tree = nltk.Tree.fromstring(str(sent_constituency))
                constituents.extend(Tree.factorize(tree))

            prons = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in enumerate(doc)]
            collected_scene.append({
                "constituent": constituents,
                "utterance": utt,
                "pron": prons
            })
        except:
            collected_scene.append({"constituent": [], "utterance": utt, "pron": []})
            print(utt)

    # # Perform Parsing and Collect Results
    # collected_scene = []
    # for utt in fa_utts:
    #     try:
    #         doc = nlp(utt)
    #         constituents = []
    #         for sent_constituency in doc._.constituency:
    #             tree = nltk.Tree.fromstring(str(sent_constituency))
    #             constituents.extend(Tree.factorize(tree))
    #
    #         prons = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in enumerate(doc)]
    #     except:
    #         collected_scene.append({"constituent": [], "utterance": utt, "pron": []})
    #         print(utt)
    #
    #     collected_scene.append({
    #         "constituent": constituents,
    #         "utterance": utt,
    #         "pron": prons
    #     })


        # if utt in [" ", "", '']:
        #     collected_scene.append({"constituent": [],"utterance": utt,"pron": []})
        #     continue
        # try:
        #     doc = nlp(utt)
        # except:
        #     collected_scene.append({"constituent": [], "utterance": utt, "pron": []})
        #     pass
        # # Perform Constituency Parsing
        # constituents = []
        # for sent_constituency in doc._.constituency:
        #     tree = nltk.Tree.fromstring(str(sent_constituency))
        #     constituents.extend(Tree.factorize(tree))
        #
        # prons = [(item.text, i, i + 1, item.pos_, item.tag_) for i, item in enumerate(doc)]
        # collected_scene.append({
        #     "constituent": constituents,
        #     "utterance": utt,
        #     "pron": prons
        # })
    with open('data/parsed_data/fa_check/' + scene_id + ".pkl", 'wb') as f:
        pkl.dump(collected_scene, f)
