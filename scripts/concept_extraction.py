import requests
import base64
from tqdm import tqdm
import time
import openai
import pickle
from tqdm import tqdm
import argparse
openai.api_key = ""
openai.api_base = ""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    args = parser.parse_args()
    data = args.data

    with open(f"file/{data}/caption_llava.pkl", "rb") as f:
        d = pickle.load(f)
    with open(f"file/{data}/rank.pkl", "rb") as f:
        r = pickle.load(f)
    thre = len(r.keys())
    candidates_dict = {}

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

    for neuro in tqdm(range(thre)):
        if neuro in candidates_dict:
            continue
        try:
            group_a = []
            for i in r[neuro][:20]:
                group_a.append(d[i])
            group_a = '\n    Description: '.join(group_a)
            text = f"    Description: {group_a}"
            prompts = f"""
            You are a data scientist . We ’ re studying neurons in an image neural network , where each neuron detects specific concepts in images . The neural network is trained to conduct {tasks_des} . I ’ ve identified the images that most strongly activate a particular neuron and will provide you with their associated text descriptions of the image . Your task is to analyze these descriptions and determine the common concept that this neuron is detecting .
    
            To arrive at the most accurate and precise explanation of what this neuron is detecting , you must engage in explicit chain of thought reasoning . Begin by thoroughly examining all provided image captions , noting any patterns or commonalities . Pay close attention to recurring terminology , described structures , and consistent visual features . Consider how these elements might interrelate to form a singular , distinctive concept that the neuron could be identifying . Evaluate the context of image and consider which aspects would be most significant or unique within this classification task .
    
            As you progress through your analysis , verbalize your thought process . Explain each step of your reasoning , from initial observations to intermediate conclusions , and finally to your overall assessment . This chain of thought approach will help ensure a comprehensive and well - reasoned final explanation .
    
            After this detailed analytical process , formulate a single , specific explanation of what the neuron is detecting . Your explanation should be as precise and fine - grained as possible , avoiding vague or general statements . Focus on specific concept or combinations of concepts . Refrain from explaining the bird species itself ( e . g ., avoid statements like " This feature represents X, which is characterized by ...") . Base your explanation solely on the information provided in the reports , without additional domain knowledge that might not be captured by the neuron .
    
            Come up with 10 distinct concepts that are more likely to be associated with the neuron . Please write a list of captions ( separated by bullet points  " * " ) .  For example :
    
                * "a dog next to a horse"
                * "a car in the rain"
                * "low quality"
                * "cars from a side view"
                * "people in a intricate dress"
                * "a joyful atmosphere"
    
            The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
                * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
                * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
                * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
                * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
                * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
                * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
                * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"
    
            Below are the image descriptions , listed in order of how strongly they activate the neuron . Use these to inform your analysis and final explanation :
    
    
            {text}        """

            chat_completion = openai.ChatCompletion.create(
                model="deepseek-chat",  # "deepseek-reasoner", "gpt-4o"
                messages=[{"role": "user", "content": prompts}])

            cot_explanation = chat_completion.choices[0].message.content

            prompt2 = f'''
            summarize the chain of thought analysis and concisely output 10 concept it detected .
    
            Please write a list of captions (separated by bullet points "*"). For example:
                    * "a dog next to a horse"
                    * "a car in the rain"
                    * "low quality"
                    * "cars from a side view"
                    * "people in a intricate dress"
                    * "a joyful atmosphere"
    
            Please only output the 10 concepts without any other text. Here is the analysis:
    
            {cot_explanation}
            '''

            chat_completion = openai.ChatCompletion.create(
                model="deepseek-chat",  # "deepseek-reasoner", "gpt-4o"
                messages=[{"role": "user", "content": prompt2}])

            candidates = chat_completion.choices[0].message.content.strip("\n").split("\n")
            candidates = [i.strip(" ").strip("*").strip("-").strip(" ").strip("\"").strip("\'").strip("*") for i in
                          candidates]
            candidates_dict[neuro] = candidates
        except:
            break

    with open(f"file/{data}/concept.pkl", "wb") as f:
        pickle.dump(candidates_dict, f)