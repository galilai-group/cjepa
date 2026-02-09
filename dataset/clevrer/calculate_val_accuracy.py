import glob
import json

answer_dir = 'dataset/clevrer/questions/val.json'
val_dir = 'val_file2'

with open(answer_dir, 'r') as f:
    answers = json.load(f)
val_files = glob.glob(val_dir + '/*.json')

for f in val_files:
    filename = f.split('/')[-1].split('.')[0] + '_acc.txt'
    dest = f'{val_dir}/{filename}'

    with open(f, 'r') as target_f:
        data = json.load(target_f)

    eval_dict = {
            'descriptive': {'correct':0, 'total':0}, 
            'explanatory':{'correct_per_choice':0, 'total_choice':0, 'correct_per_question':0, 'total_question':0}, 
            'predictive':{'correct_per_choice':0, 'total_choice':0, 'correct_per_question':0, 'total_question':0}, 
            'counterfactual':{'correct_per_choice':0, 'total_choice':0, 'correct_per_question':0, 'total_question':0}, 
        }
    invalid = 0
    for target, answer in zip(data, answers):
        assert int(answer['video_filename'].split('_')[-1].split('.')[0]) == target['scene_index']

        target_questions = target['questions']
        answer_questions = answer['questions']

        for t_q, a_q in zip(target_questions, answer_questions):
            if t_q['question_id'] != a_q['question_id']:
                invalid += 1
                continue
            q_type = a_q['question_type']


            if (q_type == 'predictive') or (q_type == 'explanatory') or (q_type=="counterfactual"):
                target_choices = t_q['choices']
                answer_choices = a_q['choices']
                correct = 0
                for t_ans, a_ans in zip(target_choices, answer_choices):
                    answer = a_ans['answer']
                    output = t_ans['answer']
                    eval_dict[q_type]['total_choice'] += 1
                    if answer == output:
                        correct += 1
                        eval_dict[q_type]['correct_per_choice'] += 1
                if len(target_choices) == correct :
                    eval_dict[q_type]['correct_per_question'] += 1
                eval_dict[q_type]['total_question'] += 1


            elif q_type == 'descriptive':
                # # for descriptive questions, only compare the first character
                # t_ans = t_ans[0]
                # a_ans = a_ans[0]
                answer = a_q['answer']
                output = t_q['answer']
                eval_dict[q_type]['total'] += 1
                if answer == output:
                    eval_dict[q_type]['correct'] += 1

            else:
                raise NotImplementedError(f'Unknown question type {q_type}')
            


    # print eval_dict
    with open(dest, 'w') as f:
        print(f'Evaluation results saved to {dest}')
        print('---------------------------------------')
        print('invalid Q&A pairs:', invalid)
        total_question = 0
        total_correct = 0
        for key in eval_dict:
            if key == 'descriptive':
                correct = eval_dict[key]['correct']
                total = eval_dict[key]['total']
                acc = correct / total if total > 0 else 0.0
                f.write(f'{key} accuracy: {acc*100:.2f}% ({correct}/{total})\n')
                print(f'{key} accuracy: {acc*100:.2f}% ({correct}/{total})')
                total_question += total
                total_correct += correct
            else:
                correct_per_choice = eval_dict[key]['correct_per_choice']
                total_choice = eval_dict[key]['total_choice']
                acc_choice = correct_per_choice / total_choice if total_choice > 0 else 0.0

                correct_per_question = eval_dict[key]['correct_per_question']
                total_question_local = eval_dict[key]['total_question']
                acc_question = correct_per_question / total_question_local if total_question_local > 0 else 0.0

                f.write(f'{key} accuracy per option: {acc_choice*100:.2f}% ({correct_per_choice}/{total_choice})\n')
                f.write(f'{key} accuracy per question: {acc_question*100:.2f}% ({correct_per_question}/{total_question_local})\n')
                print(f'{key} accuracy per option: {acc_choice*100:.2f}% ({correct_per_choice}/{total_choice})')
                print(f'{key} accuracy per question: {acc_question*100:.2f}% ({correct_per_question}/{total_question_local})')
                total_question += total_question_local
                total_correct += correct_per_question
        overall_acc = total_correct / total_question if total_question > 0 else 0.0
        f.write(f'Overall accuracy: {overall_acc*100:.2f}% ({total_correct}/{total_question})\n')
        print(f'Overall accuracy: {overall_acc*100:.2f}% ({total_correct}/{total_question})\n\n')


