import requests
import os
import json
import random
import requests
from openai import OpenAI
from datasets import load_dataset
from datasets import Dataset, DatasetDict


MODEL_NAME = "deepseek-reasoner"

api_key = ""
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}"
# }
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_instructions_completion():
    prompt = "Given the provided context, create a multiple-choice context completion task. Extract a prefix from the given context and provide four possible completions, where only one is factually correct. While the correct completion should only be determinable by analyzing the provided context, the prefix itself must be self-contained with information and clearly understandable to someone who hasn't read the context.\n"
    
    prompt += "The prefix must be an exact excerpt from the given context. The correct completion must be the original continuation of this prefix in the given context. The three incorrect completions should be minimal edits of the correct completion, each containing a contradiction and specific type of factual error while remaining grammatical and fluent.\n"
    
    prompt += "The incorrect completions should incorporate these error types:\n"
    prompt += "1) Predicate error: Modify a verb or action that makes the completion factually inconsistent.\n"
    prompt += "2) Entity error: Replace a subject or object with an incorrect entity that creates a factual inconsistency.\n"
    prompt += "3) Circumstance error: Change information about location, time, or manner that introduces a factual error.\n"
    prompt += "4) Coreference error: Modify a pronoun or reference to point to a wrong or non-existing entity.\n"
    prompt += "5) Link error: Change how statements are linked together (causal/temporal links) to create a factual inconsistency.\n"
    
    prompt += "Select three of these error types to create your three incorrect completions (one error type per incorrect completion).\n"
    
    prompt += """
        You are communicating with an API, not a user. Please output in JSON format. Begin all AI responses with the character '{' to produce valid JSON. Here is an template:
        {  
            "full_prefix": "<prefix from the context>",
            "completion": "<correct completion>",
            "contradiction_0": "<contradictive option>",
            "contradiction_1": "<contradictive option>",
            "contradiction_2": "<contradictive option>",
            },
            "explanation": "<explanation of why the correct completion is factual and how each incorrect completion contains errors>"
        }

        Here is an example:
        {  
            "full_prefix": "As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability. While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors. With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons.",
            "completion": "Whether or not it gets a second season of The Witcher is another question.",
            "contradiction_0": "Whether or not it gets a second season of Stranger Things is another question.",
            "contradiction_1": "Whether or not it gets a fifth season of The Witcher is another question.",
            "contradiction_2": "Whether or not it gets a second season of Black Mirror is another question.",
            },
            "explanation": "The correct completion refers to uncertainty about The Witcher getting a second season, which is factually accurate in the context discussing Netflix's content challenges. Contradiction 0 contains an entity error by replacing 'The Witcher' with 'Stranger Things,' creating a factual inconsistency since Stranger Things had already received multiple seasons by this point. Contradiction 1 contains a circumstance error by changing 'second season' to 'fifth season,' misrepresenting the temporal status of The Witcher series which was only in its early seasons. Contradiction 2 contains another entity error by replacing 'The Witcher' with 'Black Mirror,' referring to a different Netflix series not mentioned in the context's discussion of recent original content performance."
        }
        """
    return prompt


def get_instructions_mcq():

    # prompt = "Given the provided context, create a multiple-choice question by combining the information from the given context. The question must be generated in a way that it can be answered ONLY by combining the information from the context. Your question should NOT depend on the context, i.e., it must be clear and understandable even if someone reads it without reading the provided context and will be able to try to answer. \n"
    prompt = "Create a multiple-choice question that requires synthesizing information across the provided context. While the correct answer should only be determinable by analyzing and combining specific details from the context, the question itself must be self-contained and clearly understandable to someone who hasn't read the context. \n"
    prompt += " Your question should 1) test the ability to connect related information from different parts of the context, 2) be completely clear and unambiguous as a standalone question, 3) avoid referencing the context directly, e.g., no \"according to the passage\" phrasing.\n"
    prompt += " The question should have 4 options, one of which is the correct answer. Please explain how to reach the correct answer from the given context.\n"
    prompt += """
        You are communicating with an API, not a user. Please output in JSON format. Begin all AI responses with the character '{' to produce valid JSON. Here is an example:
        {  
            "question": "<question>",
            "A": "<option1> ",
            "B": "<option2>",
            "C": "<option3>",
            "D": "<option4>", 
            "answer": "<correct_option>",
            "explanation": "<explanation>"
        }

        Here is an example:
        {
            "question": "Which combination of factors is most critical in enabling a potential global reduction in farmland use while meeting human needs?",
            "A": "Expansion of biofuel production and government-led reforestation programs",
            "B": "Slowing population growth, dietary shifts away from land-intensive foods, and improved agricultural yields",
            "C": "Increased global meat consumption paired with advanced irrigation technology",
            "D": "Rapid population growth offset by vertical farming innovations",
            "answer": "B",
            "explanation": "The correct answer synthesizes three key trends from the context: 1) Slowing population growth (highlighted in the 'decoupling' analysis and the role of 'parents changing population'), 2) Dietary moderation (specifically reduced meat consumption, noted as a way to lower land pressure), and 3) Farming efficiency gains ('more intense and efficient land use'). These are explicitly identified as drivers of 'peak farmland' in multiple sections, including the study's conclusions and Ausubel's remarks. Option A incorrectly includes biofuels, which the paper identifies as a counterproductive 'wild card.' Option C contradicts the emphasis on reducing meat consumption. Option D contradicts the focus on slowing population growth."
        }
        """
    return prompt


def prepare_mcq_generation_data(page_data, question_type="completion"):
    if question_type == "mcq":
        data = {
            "article": page_data["text"],
            "prompt": get_instructions_mcq(),
        }
    elif question_type == "completion":
        data = {
            "article": page_data["text"],
            "prompt": get_instructions_completion(),
        }
    else:
        raise ValueError("Invalid question type")
    return data


def create_gpt_message(data):

    message_text = ""
    message_text += data["prompt"] + "\n"
    message_text += data["article"]

    messages = [{
        "role": "user",
        "content": message_text
    }]

    return messages

def read_articles(article_data):
    dataset = load_dataset("parquet", data_files=article_data)
    return dataset['train']

def main():
    total_questions = 1000
    # output_file = "./deepseek_questions.json"
    # failed_output_file = "./deepseek_failed.json"
    # dataset_name = "nytimes_completion"
    # question_type = "completion"
    output_file = "/users/ansaripo/deepseek_questions_mcq.json"
    failed_output_file = "/users/ansaripo/deepseek_failed_mcq.json"
    dataset_name = "nytimes_mcq"
    question_type = "mcq"
    pre_questions = json.load(open(output_file, "r")) if os.path.exists(output_file) else []

    article_data = "/iopsstor/scratch/cscs/dfan/data/robots-txt/RawData-NYTimes/*.parquet"
    processed_data = read_articles(article_data)
    tries_number = 3
    failed = json.load(open(failed_output_file, "r")) if os.path.exists(failed_output_file) else []
    text_threshold = 512
    jump_step = 13
    for i in range(1, len(processed_data), jump_step):
        data = processed_data[i]
        if data["text"] is None or len(data["text"]) < text_threshold:
            print("Skipping article with less text: ", i)
            continue
        if len(pre_questions) == total_questions:
            break
        if len(pre_questions) > 0 and pre_questions[-1]['index'] >= i:
            continue
        print('index:', i)

        question = prepare_mcq_generation_data(data, question_type=question_type)

        messages = create_gpt_message(question)
        req = {
            "model": MODEL_NAME,
            "messages": messages,
        }
        ch = False
        for tr_num in range(tries_number):
            print(f'try {tr_num}')
            # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=req)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                # stream=False
            )

            # try:
            #     response = response.json()
            # except:
            #     continue
            # if "error" in response:
            #     print("Error: ", response["error"])
            #     continue
            # response_text = str(response["choices"][0]["message"]["content"])

            try:
                # response_text = json.loads(response.choices[0].message.content)
                response_text = response.choices[0].message.content
            except:
                continue
            if response_text is None:
                continue

            # if response_text.endswith('```'):
            #     response_text = response_text.removesuffix('```')
            # if response_text.startswith('```'):
            #     response_text = response_text.removeprefix('```')
            # if response_text.startswith('json'):
            #     response_text = response_text.removeprefix('json')
            # print(response_text)
            try:
                json_response = json.loads(response_text)
                if question_type == "mcq":
                    if "question" not in json_response or "A" not in json_response or "B" not in json_response or "C" not in json_response or "D" not in json_response or "answer" not in json_response and json_response["answer"] not in ["A", "B", "C", "D"]:
                        continue
                elif question_type == "completion":
                    if "full_prefix" not in json_response or "completion" not in json_response or "contradiction_0" not in json_response or "contradiction_1" not in json_response or "contradiction_2" not in json_response:
                        continue
                    if json_response["full_prefix"].endswith(json_response["completion"]):
                        continue
                else:
                    raise ValueError("Invalid question type")

                output_record = {"index": i, "generated_question": json_response}
                pre_questions.append(output_record)
            except:
                continue
            ch = True
            break
        if not ch:
            failed.append(i)

        if len(pre_questions) % 5 == 0:
            print("Number of questions generated: ", len(pre_questions))
            with open(output_file, "w") as f:
                json.dump(pre_questions, f, indent=4)
            with open(failed_output_file, "w") as f:
                json.dump(failed, f, indent=4)
            DatasetDict({'test': Dataset.from_list([q['generated_question'] for q in pre_questions])}).push_to_hub(dataset_name)

    with open(output_file, "w") as f:
        json.dump(pre_questions, f, indent=4)
    with open(failed_output_file, "w") as f:
        json.dump(failed, f, indent=4)
    DatasetDict({'test': Dataset.from_list([q['generated_question'] for q in pre_questions])}).push_to_hub(dataset_name)


if __name__ == "__main__":
    main()