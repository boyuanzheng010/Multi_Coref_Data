from copy import deepcopy
import jsonlines

def merge_head_sharing_np(spans):
    temp = deepcopy(spans)
    temp.sort(key=lambda y: y[1])
    span_dict = {}
    for text, start, end in temp:
        if start in span_dict:
            span_dict[start].append([text, start, end, end - start])
        else:
            span_dict[start] = [[text, start, end, end - start]]
    spans = []
    for x in span_dict:
        spans.append(tuple([span_dict[x][0][0], span_dict[x][0][1], span_dict[x][0][2]]))
    return spans


def extract_scene_constituency(parsed_scene, scene_id):
    all_sentences = []
    all_query_spans = []
    all_utterances = []
    all_utterance_with_infos = []
    all_zh_subtitles = []

    j = 0
    for utt in parsed_scene:
        all_zh_subtitles.append(utt['utterance'])
        speaker = utt['speaker']
        sentence_tokens = [item[0] for item in utt['pron']]

        # Collect Pronouns
        pron = []
        for item in utt['pron']:
            if item[3] == "PRON":
                pron.append((item[0], item[1], item[2]))
        # Collect Noun Phrases
        noun_phrase = []
        for item in utt['constituent']:
            if "NP" in item[3]:
                noun_phrase.append((item[0], item[1], item[2]))
        noun_phrase = merge_head_sharing_np(noun_phrase)

        # Merge Pron & Noun Phrases
        mention = list(set(noun_phrase) | set(pron))
        mention.sort(key=lambda x: x[1])

        # Convert data for annotation
        all_sentences.append([speaker, ":"] + sentence_tokens)
        for span in mention:
            all_query_spans.append({
                "sentenceIndex": j,
                "startToken": span[1] + 2,
                "endToken": span[2] + 2
            })
        j += 1

    return {
        "sentences": all_sentences,
        "querySpans": all_query_spans,
        "candidateSpans": all_query_spans,
        "clickSpans": all_query_spans,
        "utterances": all_zh_subtitles,
        "scene_id": scene_id
    }


def extract_scene_constituency_chinese_align(parsed_scene, scene_id):
    all_sentences = []
    all_query_spans = []
    all_utterances = []
    all_utterance_with_infos = []
    all_zh_subtitles = []

    # Add Projected Mention Set
    projected_data_corrected = {}
    with open('data/projected_data/zh/test_corrected.chinese.512.jsonlines', 'r') as f:
        reader = jsonlines.Reader(f)
        for line in reader:
            scene_id = line['doc_key'][:10]
            projected_data_corrected[scene_id] = line
    projected_clusters = index_jsonline_clusters_as_turkle(projected_data_corrected[scene_id])
    projected_mentions = {}
    for cluster in projected_clusters:
        for mention in cluster:
            print(mention)
            sent_id, start, end = [int(number) for number in mention[0].strip().split('_')]
            if sent_id not in projected_mentions:
                projected_mentions[sent_id] = [(start, end)]
            else:
                projected_mentions[sent_id].append((start, end))

    j = 0
    for utt in parsed_scene:
        all_zh_subtitles.append(utt['utterance'])
        speaker = utt['speaker']
        # sentence_tokens = [item[0] for item in utt['pron']]
        sentence_tokens = utt['utterance']
        # Collect Pronouns
        pron = []
        for item in utt['pron']:
            if (item[2]-item[1]==1) and (item[3]=='NP'):
                pron.append((item[0], item[1], item[2]))
        # Collect Noun Phrases
        noun_phrase = []
        for item in utt['constituent']:
            if "NP" in item[3]:
                noun_phrase.append((item[0], item[1], item[2]))
        noun_phrase = merge_head_sharing_np(noun_phrase)


        # Merge Pron & Noun Phrases
        new_nps = [(item[1], item[2]) for item in noun_phrase]
        new_pron = [(item[1], item[2]) for item in pron]
        if j in projected_mentions:
            projected = projected_mentions[j]
        else:
            projected = []
        # mention = list(set([(item[1], item[2]) for item in noun_phrase]) | set([(item[1], item[2]) for item in pron]))
        # mention = list(set(new_nps)|set(new_pron)|set(projected))
        mention = list(set(new_nps)|set(new_pron))
        mention.sort(key=lambda x: x[0])

        # Convert data for annotation
        all_sentences.append([speaker, ":"] + sentence_tokens)
        for span in mention:
            all_query_spans.append({
                "sentenceIndex": j,
                "startToken": span[0] + 2,
                "endToken": span[1] + 2
            })
        j += 1

    return {
        "sentences": all_sentences,
        "querySpans": all_query_spans,
        "candidateSpans": all_query_spans,
        "clickSpans": all_query_spans,
        "utterances": all_zh_subtitles,
        "scene_id": scene_id
    }


def extract_scene_constituency_chinese_align_with_projection(parsed_scene, scene_id):
    all_sentences = []
    all_query_spans = []
    all_utterances = []
    all_utterance_with_infos = []
    all_zh_subtitles = []

    # Add Projected Mention Set
    projected_data_corrected = {}
    with open('data/projected_data/zh/test_corrected.chinese.512.jsonlines', 'r') as f:
        reader = jsonlines.Reader(f)
        for line in reader:
            scene_id = line['doc_key'][:10]
            projected_data_corrected[scene_id] = line
    projected_clusters = index_jsonline_clusters_as_turkle(projected_data_corrected[scene_id])
    projected_mentions = {}
    for cluster in projected_clusters:
        for mention in cluster:
            print(mention)
            sent_id, start, end = [int(number) for number in mention[0].strip().split('_')]
            if sent_id not in projected_mentions:
                projected_mentions[sent_id] = [(start, end)]
            else:
                projected_mentions[sent_id].append((start, end))

    j = 0
    for utt in parsed_scene:
        all_zh_subtitles.append(utt['utterance'])
        speaker = utt['speaker']
        # sentence_tokens = [item[0] for item in utt['pron']]
        sentence_tokens = utt['utterance']
        # Collect Pronouns
        pron = []
        for item in utt['pron']:
            if (item[2]-item[1]==1) and (item[3]=='NP'):
                pron.append((item[0], item[1], item[2]))
        # Collect Noun Phrases
        noun_phrase = []
        for item in utt['constituent']:
            if "NP" in item[3]:
                noun_phrase.append((item[0], item[1], item[2]))
        noun_phrase = merge_head_sharing_np(noun_phrase)


        # Merge Pron & Noun Phrases
        new_nps = [(item[1], item[2]) for item in noun_phrase]
        new_pron = [(item[1], item[2]) for item in pron]
        if j in projected_mentions:
            projected = projected_mentions[j]
        else:
            projected = []
        # mention = list(set([(item[1], item[2]) for item in noun_phrase]) | set([(item[1], item[2]) for item in pron]))
        mention = list(set(new_nps)|set(new_pron)|set(projected))
        mention.sort(key=lambda x: x[0])

        # Convert data for annotation
        all_sentences.append([speaker, ":"] + sentence_tokens)
        for span in mention:
            all_query_spans.append({
                "sentenceIndex": j,
                "startToken": span[0] + 2,
                "endToken": span[1] + 2
            })
        j += 1

    return {
        "sentences": all_sentences,
        "querySpans": all_query_spans,
        "candidateSpans": all_query_spans,
        "clickSpans": all_query_spans,
        "utterances": all_zh_subtitles,
        "scene_id": scene_id
    }

def extract_scene_constituency_farsi(parsed_scene, scene_id):
    all_sentences = []
    all_query_spans = []
    all_utterances = []
    all_utterance_with_infos = []
    all_zh_subtitles = []

    j = 0
    for utt in parsed_scene:
        all_zh_subtitles.append(utt['utterance'])
        speaker = utt['speaker']
        sentence_tokens = [item[0] for item in utt['pron']]

        # Collect Pronouns
        pron = []
        for item in utt['pron']:
            if item[3] == "PRON":
                pron.append((item[0], item[1], item[2]))
        # Collect Noun Phrases
        noun_phrase = []
        for item in utt['constituent']:
            if "NP" in item[3]:
                noun_phrase.append((item[0], item[1], item[2]))
        noun_phrase = merge_head_sharing_np(noun_phrase)

        # Merge Pron & Noun Phrases
        mention = list(set(noun_phrase) | set(pron))
        mention.sort(key=lambda x: x[1])

        # Convert data for annotation
        all_sentences.append([speaker, ":"] + list(reversed(sentence_tokens)))
        for span in mention:
            all_query_spans.append({
                "sentenceIndex": j,
                "startToken": 2 + (len(sentence_tokens)-span[2]),
                "endToken": 2 + (len(sentence_tokens)-span[1])
            })
        j += 1

    return {
        "sentences": all_sentences,
        "querySpans": all_query_spans,
        "candidateSpans": all_query_spans,
        "clickSpans": all_query_spans,
        "utterances": all_zh_subtitles,
        "scene_id": scene_id
    }


def index_jsonline_speaker_as_turkle(line):
    tokens = line['tokens']

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
    return speaker_list

def index_jsonline_utterance_as_turkle(line):
    """
    Index Utterance (Subword-level) to token-level as Turkle does
    """
    tokens = line['tokens']

    # Collect Token-Level Utterances
    utt_idxs = {}
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
    for sent_idx in utt_idxs:
        utt_idxs[sent_idx] = sorted(list(set(utt_idxs[sent_idx])))

    utterances = []
    for sent_idx in utt_idxs:
        temp_sent = []
        for idx in utt_idxs[sent_idx]:
            temp_sent.append(tokens[idx])
        utterances.append(temp_sent)
    return utterances, utt_idxs



def index_jsonline_clusters_as_turkle(line):
    """
    Turn Subtoken-indexed cluster to token indexed cluster
    """
    tokens = line['tokens']
    sentence_map = line['sentence_map']
    subtoken_map = line['subtoken_map']
    clusters = line['clusters']
    speaker_list = index_jsonline_speaker_as_turkle(line)
    _, utt_idxs = index_jsonline_utterance_as_turkle(line)

    # Turn Subtoken-level cluster index to token-level
    new_clusters = []
    for cluster in clusters:
        temp_cluster = []
        for start, end in cluster:
            sent_idx = sentence_map[start]
            w_start = subtoken_map[start]
            w_end = subtoken_map[end]
            sentence_w_indexs = utt_idxs[sent_idx]
            sent_start = sentence_w_indexs.index(w_start)
            sent_end = sentence_w_indexs.index(w_end)
            new_mention = (sent_idx, sent_start+2, sent_end+2+1)
            temp_cluster.append(new_mention)
        new_clusters.append(temp_cluster)

    output_clusters = []
    for cluster in new_clusters:
        temp_output = set()
        for mention in cluster:
            temp_output.add(tuple(["_".join([str(number) for number in mention])]))
        output_clusters.append(temp_output)

    return output_clusters











