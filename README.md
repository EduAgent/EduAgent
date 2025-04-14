### EduAgent: Generative Student Agents in Learning


#### Dataset

- Course Materials: transcript_map.py, course_material_slide.csv
- Student Demographics: student_demo.csv (for EduAgent310 dataset), student_demo_generated.csv (for EduAgent705 dataset), student_demo_config.py for persona generation
- Course AOI: aoi_material_ext_slide.csv
- Gaze, Motor Behaviors, Cognitive States: during_behavior_slide.csv
- Question Answering: student_answer_item_revised.csv, student_question.csv

##### To run the simulation

- Be sure to set your own OpenAI API key (line 18-19), Gemini API key (line 1198), and HuggingFace API key (line 1161) in agent_model_run.py
- Change configuration dictionary (line 1651) according to your own settings in agent_model_run.py
- run agent_model_run.py




#### Warning

- Due to the massive contextual data as input, each agent simulation will cost about 0.2 USD with GPT 4 and 0.02 USD with GPT 3.5. Running the whole N = 310 agents for experiment one or N = 705 agents for experiment two will result in high cost. Therefore, be sure to set your suitable agent number.


#### TODO

- Some variables in the codes and datasets are not exactly the same as depicted in the paper. We will update them soon.
- We are optimizing the codes so that it is easier to understand the code structures.


##### Please cite us if you find this repo is useful
```bibtex
@article{xu2024eduagent,
  title={Eduagent: Generative student agents in learning},
  author={Xu, Songlin and Zhang, Xinyu and Qin, Lianhui},
  journal={arXiv preprint arXiv:2404.07963},
  year={2024}
}
```
