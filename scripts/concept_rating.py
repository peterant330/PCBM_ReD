import openai
import pickle
import argparse
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    args = parser.parse_args()
    data = args.data

    openai.api_key = ""
    openai.api_base = ""

    with open(f"file/{data}/concept.pkl", "rb") as f:
        d = pickle.load(f)
    concept_list = []
    for i in list(d.keys()):
        concept_list.extend(d[i])

    scores_dict = {}

    if data == "ucf":
        tasks_des = "action classification"
    elif data == "ham":
        tasks_des = "skin lesion classification"
    elif data in ["imagenet", "cifar10", "cifar100"]:
        tasks_des = "general object recognition"
    elif data == "food":
        tasks_des = "food category classification"
    elif data == "RESISC":
        tasks_des = "scene class classification"
    elif data == "aircraft":
        tasks_des = "aircraft model classification"
    elif data == "dtd":
        tasks_des = "texture classification"
    elif data == "cub":
        tasks_des = "bird species classification"
    elif data == "flower":
        tasks_des = "flower species classification"

    for i in tqdm(d):
        if i in scores_dict:
            continue
        try:
            concepts = d[i]
            concepts_text = "\n".join(concepts)
            prompt = f'''
            I have extracted some concepts for bird species classification. Here are the candidates:
    
            {concepts_text}
    
            I want you to score these concepts according to their relavency to the task of {tasks_des}. A good concept should describe detailed characteristics, and be discriminative among different classes. The feature should also be subjective and visually identifiable. Also try to avoid use shortcut features such as background and objective judgement. 
    
            Each concept receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    
            Please first output a single line containing only ten values indicating the scores for each concept, respectively. The scores are separated by a space.
    
            In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
            '''
            chat_completion = openai.ChatCompletion.create(
                model="deepseek-reasoner", #"deepseek-reasoner", "gpt-4o"
                messages=[{"role": "user", "content": prompt}])

            rating = chat_completion.choices[0].message.content.strip("\n").split("\n")[0].split(" ")
            scores_dict[i] = rating
        except:
            continue

    with open(f"file/{data}/concept_rating.pkl", "wb") as f:
        pickle.dump(scores_dict, f)