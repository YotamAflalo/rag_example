import json
from qna_bot import load_vector_store, answer_query,model


def generete_eval_data(vector_store_path = "vectors/vector_store_data_md_spliter.json",eval_data_path = "evaluation_data/chanks_recall_questions.json",eval_raw_path = "evaluation_data/results/chanks_recall_questions_raw.json"):
    vector_store = load_vector_store(vector_store_path)

    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    results = []
    for item in eval_data:
        question = item['question']
        expected_answer = item['answer']
        expected_cunks = item['rellevent_chanks_id']
        print(f"Evaluating question: {question}")
        eval_result = answer_query(question, vector_store, model =model, eval=True)
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "expected_chunks": expected_cunks,
            "retrieved_context": eval_result.get("context", ""),
            "model_answer": eval_result.get("results", ""),
            "retrieved_chunks": eval_result.get("ids", [])
        })

    with open(eval_raw_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results
def evaluate_recall(results):
    """Evaluate recall based on the retrieved context and expected chunks."""
    total_questions = len(results)
    basic_recall_score = 0
    correct_retrievals = 0
    for item in results:
        right_retrivals = 0
        retrieved_chunks = item['retrieved_chunks']
        expected_chunks = item['expected_chunks']
        
        if set(expected_chunks).issubset(set(retrieved_chunks)):
            correct_retrievals += 1
            item['full_recall'] = 1
        else:
            item['full_recall'] = 0
        for chunk in retrieved_chunks:
            if chunk in expected_chunks:
                right_retrivals += 1
        if right_retrivals > 0:
                basic_recall_score += 1
                item['basic_recall_score'] = 1
                item['precision_score'] = right_retrivals / len(expected_chunks) if len(expected_chunks) > 0 else 0
        else:
            item['basic_recall_score'] = 0
            item['precision_score'] = 0

    recall_score = basic_recall_score / total_questions if total_questions > 0 else 0
    return results, recall_score
def evaluate_ground_truth(results,model):
    """Evaluate the model's answers against the expected answers."""
    for item in results:
        model_answer = item['model_answer'].strip().lower()
        chanks = item['retrieved_context']
        prompt = f"""your task is to evaluate the groundness of the model's answer based on the retrieved context.
        you will be given the retrieved context and the model's answer, you need to determine if the model's answer is correct based on the retrieved context.
        your response should be in the following json format:
         "thinking": the reasoning process behind your evaluation,
        "score": a score between 1 to 5, where 1 means the answer is completely not based on the context
        and 5 means the answer is completely grounded in the context. 

        the context:\n{chanks} \n\n
        the model answer : {model_answer}\n\n

        your response:
        """
        evaluation = model.invoke([
            {"role": "system", "content": prompt}
        ]).content
        try:
            evaluation_json = json.loads(evaluation)
            item['ground_truth_evaluation'] = evaluation_json
        except json.JSONDecodeError:
            item['ground_truth_evaluation'] = {"thinking": "Failed to parse evaluation", "score": 0}

    return results
def evaluate_correctness(results,model):
    """Evaluate the correctness of the model's answers against the expected answers."""
    for item in results:
        model_answer = item['model_answer'].strip().lower()
        expected_answer = item['expected_answer'].strip().lower()
        prompt = f"""your task is to evaluate the correctness of the model's answer based on the expected answer.
        you will be given the expected answer and the model's answer, you need to determine if the model's answer is correct based on the expected answer.
        your response should be in the following json format:
        "thinking": the reasoning process behind your evaluation,
        "score": a score between 1 to 5, where 1 means the answer is completely incorrect
        and 5 means the answer is completely correct. 

        the expected answer:\n{expected_answer} \n\n
        the model answer : {model_answer}\n\n

        your response:
        """
        evaluation = model.invoke([
            {"role": "system", "content": prompt}
        ]).content
        try:
            evaluation_json = json.loads(evaluation)
            item['correctness_evaluation'] = evaluation_json
        except json.JSONDecodeError:
            item['correctness_evaluation'] = {"thinking": "Failed to parse evaluation", "score": 0}

    return results

# TODO - aggregetions of the scores, and then try diffrent approaches to improve the scores, like diffrent chunking, diffrent emmbeding model, reranking
# better prompting, diffrent k values ect.
def aggregate_scores(results):
    total_questions = len(results)
    total_ground_truth_score = sum(item['ground_truth_evaluation']['score'] for item in results if 'ground_truth_evaluation' in item)
    total_correctness_score = sum(item['correctness_evaluation']['score'] for item in results if 'correctness_evaluation' in item)
    
    average_ground_truth_score = total_ground_truth_score / total_questions if total_questions > 0 else 0
    average_correctness_score = total_correctness_score / total_questions if total_questions > 0 else 0
    average_full_recall = sum(item['full_recall'] for item in results) / total_questions if total_questions > 0 else 0
    average_basic_recall = sum(item['basic_recall_score'] for item in results) / total_questions if total_questions > 0 else 0
    average_precision = sum(item['precision_score'] for item in results) / total_questions if total_questions > 0 else 0
    print(f"Average Ground Truth Score: {average_ground_truth_score}")
    print(f"Average Correctness Score: {average_correctness_score}")    
    print(f"Average Full Recall: {average_full_recall}")
    print(f"Average Basic Recall: {average_basic_recall}")
    print(f"Average Precision: {average_precision}")
    return {
        "average_ground_truth_score": average_ground_truth_score,
        "average_correctness_score": average_correctness_score,
        "average_full_recall": average_full_recall,
        "average_basic_recall": average_basic_recall,
        "average_precision": average_precision
    }

if __name__ == "__main__":
    results = generete_eval_data()
    print("Evaluating recall...")
    results, recall_score = evaluate_recall(results)
    print(f"Recall Score: {recall_score}")
    
    print("Evaluating ground truth...")
    results = evaluate_ground_truth(results,model)
    print(f"example evaluation: {results[0]['ground_truth_evaluation']}")
    print("Evaluating correctness...")
    results = evaluate_correctness(results,model)
    print(f"example evaluation: {results[0]['correctness_evaluation']}")
    with open("evaluation_data/results/chanks_recall_questions_evaluated.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # with open("evaluation_data/results/chanks_recall_questions_evaluated.json", 'r', encoding='utf-8') as f:
    #     results = json.load(f)
    aggregetions = aggregate_scores(results)
    