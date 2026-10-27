import spacy
from rfdetr.assets.coco_classes import COCO_CLASSES, COCO_CLASS_NAMES
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")

def extract_object_details_from_input(text):
  doc = nlp(text)

  object_list = []
  for d in doc:
    if(d.pos_ == "NOUN" and d.text.lower() != "style"):
      object_list.append(d.text)

  return object_list

from rfdetr.assets.coco_classes import COCO_CLASSES, COCO_CLASS_NAMES
from sentence_transformers import SentenceTransformer

def get_similar_coco_labels(entity_list):
  embed_cmp_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

  embed_user = embed_cmp_model.encode(entity_list)
  embed_coco = embed_cmp_model.encode(COCO_CLASS_NAMES)

  similarities = embed_cmp_model.similarity(embed_user, embed_coco)

  coco_sim_dict = {}
  for idx, word in enumerate(entity_list):
    coco_sim_dict[word] = {label:similarities[idx][j].item() for j, label in enumerate(COCO_CLASS_NAMES)}

    coco_sim_dict[word] = dict(sorted(coco_sim_dict[word].items(), key=lambda x: x[1], reverse=True))
  return [list(coco_sim_dict[val].items())[0][0] for val in entity_list]


text = "style this cat and dog"
print(extract_object_details_from_input(text))

user_sentences = ['woman', 'table', 'leopard', 'ball']
closest_labels = [val[0] for val in get_similar_coco_labels(user_sentences)]

print(closest_labels)