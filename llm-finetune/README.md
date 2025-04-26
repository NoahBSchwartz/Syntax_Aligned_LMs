# LLM Finetune

## Results 
We tested the fine-tuned models against SPIDER's test dataset. Below are the results for the best-performing models.
- QWEN2-7B

## Performance of model
| Model Type | Avg. %age similarity | %age semantically Valid |
|------------|------------|---------------------|
| Finetuned QWEN | 82     | 26.9                |

<!-- ## Results
We tested all SQL fine-tuned models against SPIDER's validation dataset and compared their performance with a SQL fine-tuned version of WizardCoder, the previous best-performing CUPLV model. (Note: No TSQL fine-tuned models were tested yet, coming soon.)
- Flan-T5 (248M): we changed the temperature value and enabled or disabled beam search. Below is the performance with the optimal settings.
- QWEN-2-7B-Instruct (7B): coming soon
- WizardCoder (15B)

#### Performance by Query Difficulty
| Model Type                        | Easy Queries | Medium Queries | Hard Queries | Extra Hard Queries | All Queries |
|-----------------------------------|--------------|----------------|--------------|--------------------|-------------|
| **No Fine Tuning Flan-T5-Base**   | 0%           | 0%             | 0%           | 0%                 | 0%          |
| **Finetuned-Flan-T5-Base (Temp = .45)** | 15.7%        | 4%             | 0%           | 0%                 | 5.5%        |
| **Finetuned-Flan-T5-Base**<br>(NumBeams=5, Temp = .7) | 19.4%        | 11.2%         | 12.1%         | 4.8%               | 12.3%       |
| **Finetuned-Wizard-Coder-Natsql** | 70.2%        | 63.9%          | 46%          | 34.9%              | 57.7%       |

#### Example Correct Queries

- **Query 1 (Flan-T5 performance)**:
  - Desired: `SELECT T2.name, T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.Year = 2014 INTERSECT SELECT T2.name, T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.Year = 2015`
  - Predicted: `SELECT t1.name, t1.location FROM stadium AS t1 JOIN concert AS t2 ON t1.stadium_id = t2.stadium_id WHERE t2.year = 2014 INTERSECT SELECT t1.name, t1.location FROM stadium AS t1 JOIN concert AS t2 ON t1.stadium_id = t2.stadium_id WHERE t2.year = 2015`

- **Query 2 (Flan-T5 performance)**:
  - Desired: `SELECT t1.name FROM employee AS t1 JOIN evaluation AS t2 ON t1.Employee_ID = t2.Employee_ID GROUP BY t2.Employee_ID ORDER BY count(*) DESC LIMIT 1`
  - Predicted: `SELECT t1.name FROM employee AS t1 JOIN evaluation AS t2 ON t1.employee_id = t2.employee_id GROUP BY t2.employee_id ORDER BY COUNT(*) DESC LIMIT 1`

#### Example Incorrect Queries

- **Query 1 (Flan-T5 performance)**:
  - Desired: `SELECT t1.name FROM employee AS t1 JOIN evaluation AS t2 ON t1.Employee_ID = t2.Employee_ID ORDER BY t2.bonus DESC LIMIT 1`
  - Predicted: `SELECT t1.name FROM evaluation AS t1 JOIN employee AS t2 ON t1.employee_id = t2.employee_id GROUP BY t2.employee_id ORDER BY COUNT(*) DESC LIMIT 1`

- **Query 2 (Flan-T5 performance)**:
  - Desired: `SELECT version_number, template_type_code FROM Templates WHERE version_number > 5`
  - Predicted: `SELECT t1.template_type_code, COUNT(*) FROM templates AS t1 JOIN ref_template_types AS t2 ON t1.template_type_code = t2.template_type_code GROUP BY t1.template_type_code` -->
