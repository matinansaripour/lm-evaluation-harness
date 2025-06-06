import requests
import os
import json
import random
import requests
from openai import OpenAI
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from huggingface_hub import login


MODEL_NAME = "deepseek-reasoner"

api_key = ""
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}"
# }
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def get_instructions(question_data, article):
    prompt = f"""Answer the following multiple choice question based on the given article by just outputting the letter of the correct answer. For example, if the correct answer is A, just output A.
    
    Article:
    {article}
    
    Question:
    {question_data["question"]}
    
    Options:
    A. {question_data["A"]}
    B. {question_data["B"]}
    C. {question_data["C"]}
    D. {question_data["D"]}
    
    Your answer:"""
    return prompt


def prepare_mcq_generation_data(page_data, question_data):
    return get_instructions(question_data, page_data["text"])


def create_gpt_message(prompt):
    messages = [{
        "role": "user",
        "content": prompt
    }]

    return messages


def read_articles(article_data):
    dataset = load_dataset("parquet", data_files=article_data)
    return dataset['train']

def read_articles_jsonl(article_data):
    data = []
    with open(article_data, "r") as f:
        for line in f:
            data.append(json.loads(line))
    # convert to dataset
    dataset = Dataset.from_list(data)
    return dataset

def main():
    hf_token = ''
    login(token=hf_token)
    output_file = "/users/ansaripo/deepseek_questions_mcq_2023_2024.json"
    output_file_eval = "/users/ansaripo/deepseek_questions_mcq_eval_2023_2024.json"
    failed_output_file = "/users/ansaripo/deepseek_failed_mcq_eval_2023_2024.json"
    dataset_name = "nytimes_mcq_2023_2024_eval_deepseek"
    pre_questions = json.load(open(output_file, "r")) if os.path.exists(output_file) else []
    final_answers = json.load(open(output_file_eval, "r")) if os.path.exists(output_file_eval) else []

    article_data = "/users/ansaripo/1000_mcq_news_23_24.jsonl"
    # processed_data = read_articles(article_data)
    processed_data = read_articles_jsonl(article_data)
    tries_number = 3
    failed = json.load(open(failed_output_file, "r")) if os.path.exists(failed_output_file) else []
    for q in pre_questions:
        print("processing: ", q['index'])
        check_ = False
        for pre in final_answers:
            if pre['index'] == q['index']:
                print("already evaluated")
                check_ = True
                break
        if check_:
            continue
        data = processed_data[int(q['index'])]
        print('processed data: ', len(final_answers))

        prompt = prepare_mcq_generation_data(data, q['generated_question'])

        messages = create_gpt_message(prompt)

        ch = False
        for tr_num in range(tries_number):
            print(f'try {tr_num}')
            # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=req)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )

            try:
                response_ = response.choices[0].message.content
                if len(response_) > 1 and response_[-1] == '.':
                    response_text = response_.strip()[-2]
                else:
                    response_text = response_.strip()[-1]
                if response_text not in ['A', 'B', 'C', 'D']:
                    response_text = response_.strip()[0]
            except:
                continue
            if response_text is None:
                continue
            print(response_text)
            if response_text not in ['A', 'B', 'C', 'D']:
                continue

            q['prediction'] = response_text
            final_answers.append(q)
            ch = True
            break
        if not ch:
            failed.append(q['index'])

        if len(final_answers) % 5 == 0:
            print("saving")
            with open(output_file_eval, "w") as f:
                json.dump(final_answers, f, indent=4)
            with open(failed_output_file, "w") as f:
                json.dump(failed, f, indent=4)
            DatasetDict({'test': Dataset.from_list(final_answers)}).push_to_hub(dataset_name)

    with open(output_file_eval, "w") as f:
        json.dump(final_answers, f, indent=4)
    with open(failed_output_file, "w") as f:
        json.dump(failed, f, indent=4)
    DatasetDict({'test': Dataset.from_list(final_answers)}).push_to_hub(dataset_name)


if __name__ == "__main__":
    main()

