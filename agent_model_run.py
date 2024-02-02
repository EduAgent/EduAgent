import os,sys,re,math,time,json,requests
from datetime import datetime
import numpy as np 
import pandas as pd 
from collections import defaultdict, Counter
from termcolor import colored, cprint
import asyncio,random
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

def default_string():
    return ''

from transcript_map import video_transcript_dict, question_id_map_slide_dict, slide_summary_dict

import openai
from openai import OpenAI
openai.api_key = ""
client = OpenAI(api_key = "")

class Avatar(object):
    # agent id should be the same as the real student's ID
    # each agent's persona, memory, and reflection will be stored into json file. The prompt/response history will be stored into a separate text file. 
    def __init__(self,agent_config,agent_id):
        self.agent_config = agent_config
        self.agent_id = agent_id
        self.persona = ''

        self.sim_config_name = ['reflection_choice','memory_component_choice','memory_source','forget_effect','sim_strategy','example_demo','gpt_type']
        self.sim_config_set = [self.agent_config[sim_item] for sim_item in self.sim_config_name]
        self.memory_component_choice_list = self.agent_config['memory_component_choice'].split('+')
        self.choice_map = {'A':0,'B':1,'C':2,'D':3,'a':0,'b':1,'c':2,'d':3}
        
        self._check_result_file()
        self._make_assertion()
        self._make_result_folder()
        self._load_dataset()
        self._init_agent_dataset()
        self._init_result_write()

    def _check_result_file(self):
        if os.path.exists(self.agent_config['result_path']+'/result_ind_dur.csv'):
            with open(self.agent_config['result_path']+'/result_ind_dur.csv', 'r') as file:
                for i,line in enumerate(file):
                    if len(line.split(',')) != 40:
                        print(i,line)
                        assert 1 == 0
        if os.path.exists(self.agent_config['result_path']+'/result_ind_post.csv'):
            with open(self.agent_config['result_path']+'/result_ind_post.csv', 'r') as file:
                for i,line in enumerate(file):
                    if len(line.split(',')) != 11:
                        print(i,line)
                        assert 1 == 0


    def _make_assertion(self):
        assert self.agent_config['memory_source'] in ['real','sim']
        assert self.agent_config['forget_effect'] in ['no_memory','random_half_plus_recent_one','all_plus_recent_one','only_recent_one']
        assert self.agent_config['reflection_choice'] in ['yes','no']
        assert self.agent_config['memory_component_choice'] in ['KM','KM+PM','KM+PM+MM','KM+PM+MM+CM','KM+PM+CM','KM+MM+CM']
        assert self.agent_config['sim_strategy'] in ['standard', 'cot', 'react', 'standard_cog', 'cot_cog', 'react_cog']
        assert self.agent_config['example_demo'] in ['yes','no']
        assert self.agent_config['gpt_type'] in [0,1,2,3,4]


    def _make_result_folder(self):
 
        
        self.root_folder = self.agent_config['result_path'] + '/' + self.agent_config['memory_source'] + '_' + self.agent_config['forget_effect'] + '_reflect-' + self.agent_config['reflection_choice'] + '_' + self.agent_config['memory_component_choice'] + '_' + self.agent_config['sim_strategy'] + '_example-' + self.agent_config['example_demo'] + '_' + str(self.agent_config['gpt_type'])

        self.log_file = self.root_folder + '/log/' + str(self.agent_id) + '.txt'
        self.agent_memory_file = self.root_folder + '/agent_memory/' + str(self.agent_id) + '.json'
        self.user_memory_file = self.root_folder + '/user_memory/' + str(self.agent_id) + '.json'

        # self.sim_result_dur_path = self.agent_config['result_path']+'/result_ind_dur.csv'
        # self.sim_result_post_path = self.agent_config['result_path']+'/result_ind_post.csv'

        self.sim_result_dur_path = self.root_folder+'/log/'+str(self.agent_id)+'_result_ind_dur.csv'
        self.sim_result_post_path = self.root_folder+'/log/'+str(self.agent_id)+'_result_ind_post.csv'

        self._store_log('\n\n Start running new simulation program '+ '='*100 +'\n\n')

    def _load_dataset(self):
        self.demo_dataset_origin = pd.read_csv(self.agent_config['dataset_path']+'/student_demo.csv')
        self.demo_dataset_extend = pd.read_csv(self.agent_config['dataset_path']+'/student_demo_generated.csv')
        self.course_dataset = pd.read_csv(self.agent_config['dataset_path']+'/course_material_slide.csv')
        self.student_answer_item_dataset = pd.read_csv(self.agent_config['dataset_path']+'/student_answer_item_revised.csv')
        # student_id,start_timestamp_transcript,end_timestamp_transcript,transcript_id,start_timestamp_gaze,end_timestamp_gaze,gaze_entropy_stationary,gaze_entropy_transition,gaze_aoi_id,gaze_aoi_center_x_ratio,gaze_aoi_center_y_ratio,valid_focus,course_follow
        self.during_dataset = pd.read_csv(self.agent_config['dataset_path']+'/during_behavior_slide.csv')
        self.aoi_material_dataset = pd.read_csv(self.agent_config['dataset_path']+'/aoi_material_ext_slide.csv',sep='\t')
        self.question_dataset = pd.read_csv(self.agent_config['dataset_path']+'/student_question.csv',sep='\t')

        self.agent_id_to_course_dict = self._get_agent_id_to_course_dict()
        self.course_to_transcript_dict = self._get_course_to_transcript_dict()
        

    def _init_agent_dataset(self):
        self.video_id = self.agent_id_to_course_dict[self.agent_id]
        self.example_id_list = self.agent_config['example_user_dict']['video_'+str(self.video_id)][0:1]

        self.transcript_id_list_all = self.course_to_transcript_dict[self.video_id]
        self.transcript_id_list_all.sort()
        self.transcript_id_list_simulation = self.transcript_id_list_all
        self.transcript_num_all = len(self.transcript_id_list_all)

        self.question_dataset_agent = self.question_dataset[self.question_dataset['course_name']==self.video_id]
        self.question_id_list = list(set(self.question_dataset_agent['question_id']))
        self.question_id_list.sort()

        self.transcript_dict_agent = video_transcript_dict['video_'+str(self.video_id)]
        self.course_dataset_agent = self.course_dataset[self.course_dataset['course_name']==self.video_id]
        self.aoi_material_dataset_agent = self.aoi_material_dataset[self.aoi_material_dataset['course_name']==self.video_id]

        self.student_answer_item_dataset_agent = self.student_answer_item_dataset[self.student_answer_item_dataset['student_id']==self.agent_id]
        self.during_dataset_agent = self.during_dataset[self.during_dataset['student_id']==self.agent_id]
        self.demo_agent = self.demo_dataset_origin[self.demo_dataset_origin['student_id']==self.agent_id]
        if len(self.demo_agent) == 0:
            self.demo_agent = self.demo_dataset_extend[self.demo_dataset_extend['student_id']==self.agent_id]
            if len(self.demo_agent) == 0: 
                print('Agent ID can not be found...')
                assert 1==0

    def _get_result_specific_config(self,result_table):
        result_table_new = result_table.copy()
        for config_name in self.sim_config_name:
            result_table_new = result_table_new[result_table_new[config_name]==self.agent_config[config_name]]
            # print(config_name,len(result_table_new))
        # print('len(result_table_new):',len(result_table_new))
        result_table_new = result_table_new[result_table_new['student_id']==self.agent_id]
        return result_table_new

    def _init_result_write(self):
        self.metric_ind_dur_list = ['gaze_aoi_accuracy','gaze_aoi_distance','motor_aoi_accuracy','motor_aoi_distance','workload_diff','curiosity_diff','valid_focus_diff','follow_ratio_diff','engagement_accuracy','confusion_accuracy']
        self.metric_ind_question_list = ['choice_similarity','accuracy_similarity']

        dur_result_list = ['gaze_aoi_id','gaze_aoi_center_tuple_x','gaze_aoi_center_tuple_y','motor_aoi_id','motor_aoi_center_tuple_x','motor_aoi_center_tuple_y','workload','curiosity','valid_focus','course_follow','engagement','confusion']
        user_dur_result_list = ['user_'+d_item for d_item in dur_result_list]
        agent_dur_result_list = ['agent_'+d_item for d_item in dur_result_list]

        if os.path.exists(self.sim_result_dur_path):
            self.exist_simulation_result_dur = pd.read_csv(self.sim_result_dur_path)
            self.exist_simulation_result_dur_specific_config = self._get_result_specific_config(self.exist_simulation_result_dur)
            # print('len(self.exist_simulation_result_dur_specific_config):',len(self.exist_simulation_result_dur_specific_config))
            self.exist_simulation_transcript_id_list = list(set(self.exist_simulation_result_dur_specific_config['transcript_id']))
            self.exist_simulation_transcript_id_list.sort()
        else:
            self.exist_simulation_result_dur = None 
            self.exist_simulation_transcript_id_list = []
            with open(self.sim_result_dur_path,'a+') as fwrite:
                fwrite.write(','.join(self.sim_config_name+['student_id','transcript_id','sentence_id']+user_dur_result_list+agent_dur_result_list+self.metric_ind_dur_list)+'\n')
        
        if os.path.exists(self.sim_result_post_path):
            self.exist_simulation_result_post = pd.read_csv(self.sim_result_post_path)
            self.exist_simulation_result_post_specific_config = self._get_result_specific_config(self.exist_simulation_result_post)
            self.exist_simulation_question_id_list = list(set(self.exist_simulation_result_post_specific_config['question_id']))
        else:
            self.exist_simulation_result_post = None 
            self.exist_simulation_question_id_list = []
            with open(self.sim_result_post_path,'a+') as fwrite:
                fwrite.write(','.join(self.sim_config_name+['student_id','question_id']+['user_answer','agent_answer','correct_answer']+self.metric_ind_question_list)+'\n')


       
    def _get_agent_id_to_course_dict(self):
        student_id_list_origin = list(set(self.demo_dataset_origin['student_id']))
        student_id_list_extend = list(set(self.demo_dataset_extend['student_id']))
        student_id_list = student_id_list_origin + student_id_list_extend
        agent_id_to_course_dict = {}
        for student_id in student_id_list:
            if student_id in student_id_list_origin:
                demo_data_item = self.demo_dataset_origin[self.demo_dataset_origin['student_id']==student_id]
            elif student_id in student_id_list_extend:
                demo_data_item = self.demo_dataset_extend[self.demo_dataset_extend['student_id']==student_id]
            else:
                assert 1==0
            agent_id_to_course_dict[student_id] = demo_data_item['video_id'].values[0]
        return agent_id_to_course_dict

    def _get_course_to_transcript_dict(self):
        course_id_list = list(set(self.course_dataset['course_name']))
        course_to_transcript_dict = {}
        for course_id in course_id_list:
            course_data_item = self.course_dataset[self.course_dataset['course_name']==course_id]
            course_to_transcript_dict[course_id] = list(set(course_data_item['slide_id_from_zero']))
            course_to_transcript_dict[course_id].sort()
        return course_to_transcript_dict

    def _get_aoi_choice_str(self,aoi_material_table):
        # course_name	start_timestamp	end_timestamp	transcript_id	aoi_id	aoi_upper_left_x	aoi_upper_left_y	aoi_lower_right_x	aoi_lower_right_y	aoi_center_x	aoi_center_y	aoi_content
        aoi_id_list = list(set(aoi_material_table['aoi_id']))
        aoi_id_list.sort()
        aoi_num = len(aoi_id_list)
        aoi_material_choice_str = f'\n There are {aoi_num} AOIs. Each AOI ID and its contents are listed below: \n'
        for aoi_id in aoi_id_list:
            aoi_material_item = aoi_material_table[aoi_material_table['aoi_id']==aoi_id]
            aoi_content = aoi_material_item['aoi_content'].values[0]
            aoi_material_choice_str += f'#AOI {aoi_id}#: {aoi_content}. \n'
        
        return aoi_material_choice_str

    def _generate_persona(self,demo_item):
        if 'attitude' in demo_item.columns:
            # video_id,student_id,age,gender,major,education,attitude,exam,focus,curiosity,interest,priors,compliance,smartness,family
            persona = ('Student #' + str(demo_item['student_id'].values[0]) + ' whose gender is ' + str(demo_item['gender'].values[0]) + ' and is ' + str(demo_item['age'].values[0]) + ' . This student education level is ' + 
                 str(demo_item['education'].values[0]) + ' in the major of ' + str(demo_item['major'].values[0]) + '. Moreover, this student is ' + 
                 str(demo_item['attitude'].values[0]) + ' in the course and ' + str(demo_item['exam'].values[0]) + '. During the course, this student is ' + 
                 str(demo_item['focus'].values[0]) + ' and is ' + str(demo_item['curiosity'].values[0]) + '. What\'s more, this student is ' +
                 str(demo_item['interest'].values[0]) + ' and has ' + str(demo_item['priors'].values[0]) + '. Additionally, this student is ' + 
                 str(demo_item['smartness'].values[0]) + ' and is ' + str(demo_item['compliance'].values[0]) + '. Finally, this student\'s ' +
                 str(demo_item['family'].values[0]) + '. \n'
                 )
        else:
            persona = ('Student #' + str(demo_item['student_id'].values[0]) + ' who is a ' + str(demo_item['gender'].values[0]) + ' ' + str(demo_item['age'].values[0]) + '-year-old ' + 
                 str(demo_item['education'].values[0]) + ' in the major of ' + str(demo_item['major'].values[0]) + '. The student\'s machine learning familarity level is ' + 
                 str(demo_item['ML_familarity'].values[0]) + ', experience level to use AI-related techniques to solve problems is ' + str(demo_item['ML_Experience'].values[0]) + 
                 ' and overall machine learning experience level rate is ' + str(demo_item['ML_bg_rate'].values[0]) + '(1: no experience, 5: expert). \n')
        return persona

    def instantiate_profile(self):
        # gender,age,major,education,ML_familarity,ML_Experience,ML_bg_rate
        self.persona = self._generate_persona(self.demo_agent)
        self._store_log('Finish instantiating persona: '+self.persona+'\n\n')

    def instantiate_memory(self):
        if os.path.exists(self.user_memory_file):
            try:
                with open(self.user_memory_file, 'r') as json_file:
                    self.agent_real_memory_stream = json.load(json_file)
            except:
                self.agent_real_memory_stream = []
                with open(self.user_memory_file, 'w') as json_file:
                    json.dump(self.agent_real_memory_stream, json_file, indent=4)
        else:
            self.agent_real_memory_stream = []
            with open(self.user_memory_file, 'w') as json_file:
                json.dump(self.agent_real_memory_stream, json_file, indent=4)

        if os.path.exists(self.agent_memory_file):
            try:
                with open(self.agent_memory_file, 'r') as json_file:
                    self.agent_sim_memory_stream = json.load(json_file)
            except:
                self.agent_sim_memory_stream = []
                with open(self.agent_memory_file, 'w') as json_file:
                    json.dump(self.agent_sim_memory_stream, json_file, indent=4)
        else:
            self.agent_sim_memory_stream = []
            with open(self.agent_memory_file, 'w') as json_file:
                json.dump(self.agent_sim_memory_stream, json_file, indent=4)
        
        self._store_log('\n\n Finish instantiating memory: '+'\n\n')


    def _get_example_demo_str_per(self,example_id,slide_id,question_id_list):
        # output: demo user persona, gaze AOI ID, motor AOI ID, cognitive states, question choice
        demo_item_example = self.demo_dataset_origin[self.demo_dataset_origin['student_id']==example_id]
        example_persona = self._generate_persona(demo_item_example)

        during_dataset_example = self.during_dataset[self.during_dataset['student_id']==example_id]
        during_item = during_dataset_example[during_dataset_example['slide_id_from_zero']==slide_id]

        sentence_id_list = list(set(during_item['transcript_id']))
        sentence_id_list.sort()
        user_gaze_aoi_id,user_gaze_aoi_center_tuple,user_motor_aoi_id,user_motor_aoi_center_tuple = {},{},{},{}
        user_workload, user_curiosity, user_valid_focus, user_course_follow, user_engagement, user_confusion = {},{},{},{},{},{}
        user_workload_str, user_curiosity_str, user_valid_focus_str, user_course_follow_str, user_engagement_str, user_confusion_str = '', '', '', '', '', ''
        user_gaze_aoi_id_str, user_motor_aoi_id_str = '', ''
        user_dur_counter = {}
        for sentence_id in sentence_id_list:
            during_item_per = during_item[during_item['transcript_id']==sentence_id]
            user_gaze_aoi_id[sentence_id], user_gaze_aoi_center_tuple[sentence_id] = self._get_real_gaze(during_item_per)
            user_motor_aoi_id[sentence_id], user_motor_aoi_center_tuple[sentence_id] = self._get_real_motor(during_item_per)
            user_workload[sentence_id], user_curiosity[sentence_id], user_valid_focus[sentence_id], user_course_follow[sentence_id], user_engagement[sentence_id], user_confusion[sentence_id] = self._get_real_cognitive_state(during_item_per)
            if user_workload[sentence_id] != None: 
                if user_workload_str == '':
                    user_workload_str += f'\n # Workload # trajectory is: {user_workload[sentence_id]}, '
                else:
                    user_workload_str += f'{user_workload[sentence_id]}, '
            if user_curiosity[sentence_id] != None: 
                if user_curiosity_str == '':
                    user_curiosity_str += f'\n # Curiosity # trajectory is: {user_curiosity[sentence_id]}, '
                else:
                    user_curiosity_str += f'{user_curiosity[sentence_id]}, '
            if user_valid_focus[sentence_id] != None: 
                if user_valid_focus_str == '':
                    user_valid_focus_str += f'\n # Valid Focus # trajectory is: {user_valid_focus[sentence_id]}, '
                else:
                    user_valid_focus_str += f'{user_valid_focus[sentence_id]}, '
            if user_course_follow[sentence_id] != None: 
                if user_course_follow_str == '':
                    user_course_follow_str += f'\n # Course Follow # trajectory is: {user_course_follow[sentence_id]}, '
                else:
                    user_course_follow_str += f'{user_course_follow[sentence_id]}, '
            if user_engagement[sentence_id] != None: 
                if user_engagement_str == '':
                    user_engagement_str += f'\n # Engagement # trajectory is: {user_engagement[sentence_id]}, '
                else:
                    user_engagement_str += f'{user_engagement[sentence_id]}, '
            if user_confusion[sentence_id] != None: 
                if user_confusion_str == '':
                    user_confusion_str += f'\n # Confusion # trajectory is: {user_confusion[sentence_id]}, '
                else:
                    user_confusion_str += f'{user_confusion[sentence_id]}, '
            
            if user_gaze_aoi_id[sentence_id] != None: 
                if user_gaze_aoi_id_str == '':
                    user_gaze_aoi_id_str += f'\n # Gaze Watch AOI # trajectory is: {user_gaze_aoi_id[sentence_id]}, '
                else:
                    user_gaze_aoi_id_str += f'{user_gaze_aoi_id[sentence_id]}, '
            
            if user_motor_aoi_id[sentence_id] != None: 
                if user_motor_aoi_id_str == '':
                    user_motor_aoi_id_str += f'\n # Mouse Move AOI # trajectory is: {user_motor_aoi_id[sentence_id]}, '
                else:
                    user_motor_aoi_id_str += f'{user_motor_aoi_id[sentence_id]}, '

        student_answer_item = self.student_answer_item_dataset[self.student_answer_item_dataset['student_id']==example_id]

        user_answer_letter = {}     
        user_answer_letter_str = ''
        question_id_list.sort()
        for question_id in question_id_list:
            user_answer_item = student_answer_item[student_answer_item['question_id']=='test_q'+str(question_id)]
            user_answer_letter[question_id] = user_answer_item['choice'].values[0]

            if user_answer_letter[question_id] != None: 
                if user_answer_letter_str == '':
                    user_answer_letter_str += f'\n # Question Choice #: Question ID: {question_id}, Student Choice: {user_answer_letter[question_id]}, '
                else:
                    user_answer_letter_str += f'Question ID: {question_id}, Student Choice: {user_answer_letter[question_id]}, '

        demo_str_per = f'\n\n # Example Student {example_id} #\n'
        demo_str_per += example_persona
        demo_str_per += f'\n In the current # slide {slide_id} #, for student {example_id}: '

        demo_str_per = demo_str_per + user_workload_str + user_curiosity_str + user_valid_focus_str + user_course_follow_str + user_engagement_str + user_confusion_str + user_gaze_aoi_id_str + user_motor_aoi_id_str + user_answer_letter_str

        return demo_str_per

    def obtain_example_demo_str(self,slide_id,question_id_list):
        if len(self.example_id_list) == 0: return ''
        demo_str_all = f'\n Here are the # examples # of performance of another student in the current slide #{slide_id}#. \n'
        for e_i,example_id in enumerate(self.example_id_list):
            demo_str_per = self._get_example_demo_str_per(example_id,slide_id,question_id_list)
            demo_str_all += demo_str_per
        demo_str_all += '\n Note that these examples could give you insights and reference about student learning behaviors but you should not directly copy their results.\n'
        return demo_str_all

    def _find_match_cognitive_gaze_motor(self,resyntax,input_string):
        pattern = re.compile(resyntax, re.IGNORECASE)
        matches = pattern.findall(input_string)
        if matches:
            match_result_list = [{'sentence_id': float(match[0]), 'workload': float(match[1]), 'curiosity': float(match[2]), 'valid_focus': float(match[3]), 'course_follow': float(match[4]), 'engagement': float(match[5]), 'confusion': float(match[6]), 'gaze_aoi_id': float(match[7]), 'motor_aoi_id': float(match[8])} for match in matches]
            return match_result_list
        else:
            return []


    def _find_match_cognitive(self,resyntax,input_string):
        pattern = re.compile(resyntax, re.IGNORECASE)
        matches = pattern.findall(input_string)
        if matches:
            match_result_list = [{'sentence_id': float(match[0]), 'workload': float(match[1]), 'curiosity': float(match[2]), 'valid_focus': float(match[3]), 'course_follow': float(match[4]), 'engagement': float(match[5]), 'confusion': float(match[6])} for match in matches]
            return match_result_list
        else:
            return []
        
    def _find_match_gaze(self,resyntax,input_string):
        pattern = re.compile(resyntax, re.IGNORECASE)
        matches = pattern.findall(input_string)
        if matches:
            match_result_list = [{'sentence_id': float(match[0]), 'gaze_aoi_id': float(match[1])} for match in matches]
            return match_result_list
        else:
            return []

    def _find_match_motor(self,resyntax,input_string):
        pattern = re.compile(resyntax, re.IGNORECASE)
        matches = pattern.findall(input_string)
        if matches:
            match_result_list = [{'sentence_id': float(match[0]), 'motor_aoi_id': float(match[1])} for match in matches]
            return match_result_list
        else:
            return []

    def _find_match_choice(self,resyntax,input_string):
        pattern = re.compile(resyntax, re.IGNORECASE)
        matches = pattern.findall(input_string)
        if matches:
            match_result_list = [{'question_id': float(match[0]), 'choice': match[1].upper()} for match in matches]
            return match_result_list
        else:
            return []

    def _extract_match_gaze(self,match_gaze_list,aoi_material_table):
        if len(match_gaze_list) != 0:
            gaze_aoi_id,gaze_center_tuple = {},{}
            for match_gaze in match_gaze_list:
                sentence_id_value = match_gaze['sentence_id']
                aoi_id_value = match_gaze['gaze_aoi_id']
                gaze_aoi_id[sentence_id_value] = aoi_id_value
                aoi_piece = aoi_material_table[aoi_material_table['aoi_id']==aoi_id_value]
                if len(aoi_piece) == 0:
                    gaze_aoi_id[sentence_id_value] = None
                    gaze_center_tuple[sentence_id_value] = (None,None)
                else:
                    gaze_aoi_center_x, gaze_aoi_center_y = aoi_piece['aoi_center_x'].values[0], aoi_piece['aoi_center_y'].values[0]
                    gaze_center_tuple[sentence_id_value] = (gaze_aoi_center_x, gaze_aoi_center_y)

            gaze_aoi_id = dict(sorted(gaze_aoi_id.items()))
            gaze_center_tuple = dict(sorted(gaze_center_tuple.items()))
        else:
            gaze_aoi_id,gaze_center_tuple = {}, {}
        
        return gaze_aoi_id,gaze_center_tuple

    def _extract_match_motor(self,match_mouse_list,aoi_material_table):
        if len(match_mouse_list) != 0:
            move_aoi_id,move_center_tuple = {},{}
            for match_mouse in match_mouse_list:
                sentence_id_value = match_mouse['sentence_id']
                aoi_id_value = match_mouse['motor_aoi_id']
                move_aoi_id[sentence_id_value] = aoi_id_value
                aoi_piece = aoi_material_table[aoi_material_table['aoi_id']==aoi_id_value]
                if len(aoi_piece) == 0:
                    move_aoi_id[sentence_id_value] = None
                    move_center_tuple[sentence_id_value] = (None,None)
                else:
                    move_aoi_center_x, move_aoi_center_y = aoi_piece['aoi_center_x'].values[0], aoi_piece['aoi_center_y'].values[0]
                    move_center_tuple[sentence_id_value] = (move_aoi_center_x, move_aoi_center_y)
            move_aoi_id = dict(sorted(move_aoi_id.items()))
            move_center_tuple = dict(sorted(move_center_tuple.items()))

        else:
            move_aoi_id,move_center_tuple = {}, {}
        
        return move_aoi_id,move_center_tuple

    def _extract_match_choice(self,match_choice_list):
        if len(match_choice_list) != 0:
            agent_choice = {}
            for match_choice in match_choice_list:
                question_id_value = match_choice['question_id']
                choice_value = match_choice['choice']
                try:
                    agent_choice[question_id_value] = self.choice_map[choice_value]
                except:
                    agent_choice[question_id_value] = None 
            agent_choice = dict(sorted(agent_choice.items()))
        else:
            agent_choice = {}

        return agent_choice
        
    def _extract_match_cognitive(self,match_cognitive_list): 
        if len(match_cognitive_list) != 0:
            agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion = {},{},{},{},{},{}
            for match_cognitive in match_cognitive_list:
                sentence_id_value = match_cognitive['sentence_id']
                agent_workload[sentence_id_value] = match_cognitive['workload']
                agent_curiosity[sentence_id_value] = match_cognitive['curiosity']
                agent_valid_focus[sentence_id_value] = match_cognitive['valid_focus']
                agent_course_follow[sentence_id_value] = match_cognitive['course_follow']
                agent_engagement[sentence_id_value] = match_cognitive['engagement']
                agent_confusion[sentence_id_value] = match_cognitive['confusion']
            agent_workload = dict(sorted(agent_workload.items()))
            agent_curiosity = dict(sorted(agent_curiosity.items()))
            agent_valid_focus = dict(sorted(agent_valid_focus.items()))
            agent_course_follow = dict(sorted(agent_course_follow.items()))
            agent_engagement = dict(sorted(agent_engagement.items()))
            agent_confusion = dict(sorted(agent_confusion.items()))
        else:
            agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion = {},{},{},{},{},{}

        return agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion

    def _extract_match_cognitive_gaze_motor(self,match_cognitive_gaze_motor_list,aoi_material_table): 
        if len(match_cognitive_gaze_motor_list) != 0:
            agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,gaze_aoi_id,gaze_center_tuple,move_aoi_id,move_center_tuple = {},{},{},{},{},{},{},{},{},{}
            for match_cognitive_gaze_motor in match_cognitive_gaze_motor_list:
                sentence_id_value = match_cognitive_gaze_motor['sentence_id']
                agent_workload[sentence_id_value] = match_cognitive_gaze_motor['workload']
                agent_curiosity[sentence_id_value] = match_cognitive_gaze_motor['curiosity']
                agent_valid_focus[sentence_id_value] = match_cognitive_gaze_motor['valid_focus']
                agent_course_follow[sentence_id_value] = match_cognitive_gaze_motor['course_follow']
                agent_engagement[sentence_id_value] = match_cognitive_gaze_motor['engagement']
                agent_confusion[sentence_id_value] = match_cognitive_gaze_motor['confusion']

                gaze_aoi_id_value = match_cognitive_gaze_motor['gaze_aoi_id']
                gaze_aoi_id[sentence_id_value] = gaze_aoi_id_value
                gaze_aoi_piece = aoi_material_table[aoi_material_table['aoi_id']==gaze_aoi_id_value]
                if len(gaze_aoi_piece) == 0:
                    gaze_aoi_id[sentence_id_value] = None
                    gaze_center_tuple[sentence_id_value] = (None,None)
                else:
                    gaze_aoi_center_x, gaze_aoi_center_y = gaze_aoi_piece['aoi_center_x'].values[0], gaze_aoi_piece['aoi_center_y'].values[0]
                    gaze_center_tuple[sentence_id_value] = (gaze_aoi_center_x, gaze_aoi_center_y)

                motor_aoi_id_value = match_cognitive_gaze_motor['motor_aoi_id']
                move_aoi_id[sentence_id_value] = motor_aoi_id_value
                motor_aoi_piece = aoi_material_table[aoi_material_table['aoi_id']==motor_aoi_id_value]
                if len(motor_aoi_piece) == 0:
                    move_aoi_id[sentence_id_value] = None
                    move_center_tuple[sentence_id_value] = (None,None)
                else:
                    move_aoi_center_x, move_aoi_center_y = motor_aoi_piece['aoi_center_x'].values[0], motor_aoi_piece['aoi_center_y'].values[0]
                    move_center_tuple[sentence_id_value] = (move_aoi_center_x, move_aoi_center_y)
            agent_workload = dict(sorted(agent_workload.items()))
            agent_curiosity = dict(sorted(agent_curiosity.items()))
            agent_valid_focus = dict(sorted(agent_valid_focus.items()))
            agent_course_follow = dict(sorted(agent_course_follow.items()))
            agent_engagement = dict(sorted(agent_engagement.items()))
            agent_confusion = dict(sorted(agent_confusion.items()))
            gaze_aoi_id = dict(sorted(gaze_aoi_id.items()))
            gaze_center_tuple = dict(sorted(gaze_center_tuple.items()))
            move_aoi_id = dict(sorted(move_aoi_id.items()))
            move_center_tuple = dict(sorted(move_center_tuple.items()))
        else:
            agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,gaze_aoi_id,gaze_center_tuple,move_aoi_id,move_center_tuple = {},{},{},{},{},{},{},{},{},{}

        return agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,gaze_aoi_id,gaze_center_tuple,move_aoi_id,move_center_tuple



    def action_gaze_mouse_cog_question_concise(self,sentence_id_list,example_demo_str,sim_strategy,retrieved_memory,transcript_id,transcript_material,aoi_material_table,question_content_dict,choice_content_dict,repeat_threshold=3, current_repetition=0):
        self._store_log('\n Start Gaze Motor Cognitive State Question Simulation '+'-'*40+'\n\n')
        sys_prompt = 'Assume you are the ' + self.persona + '.\n The information above is your # persona #. '

        task_overview = ('\n Assume you are on the online course. '
            +'Your task is to simulate the student # cognitive states # (task 1), # gaze watch AOI # (task 2), # mouse move AOI # (task 3), and # question choice # (task 4) ' 
            +'The details are depicted below. \n')

        task_detail = ('\n The slide in the course is segmented into several area of interests (AOIs) and you could either watch each AOI or use the computer mouse to explore them.' 
                +'\n Each slide contains a list of transcripts from the teacher.'
                +'\n Your task 1 is to simulate the trajectory of this student\'s six cognitive states in the transcript list in the current slide on the course, including:'
                +'\n WORKLOAD, CURIOSITY, VALID FOCUS, COURSE FOLLOW, ENGAGEMENT, CONFUSION.' 
                +'\n Each state is from 0 (very low level) to 1 (very high level) and should be accurate to at least two decimal places.' 
                +'\n Your task 2 is to simulate the trajectory of this student\'s gaze watching behaviors and indicate which specific AOI you are currently watching during each transcript.'
                +'\n Your task 3 is to simulate the trajectory of this student\'s mouse moving behavior and indicate which specific AOI you come across with your mouse during each transcript.'
                +'\n Your task 4 is to simulate this student\'s learning performance to answer each of the post-course questions.\n'
                )


        cog_str = ('\n Here are the # simulation strategy # for you to learn from history data. '
            +'\n For cognitive states simulation (task 1), you have steps below. First, consider your past cognitive states trajectories in the history because your current cognitive states may be correlated with past cognitive states.'
            +'\n Second, analyze correlations between past slide AOIs and past gaze/motor trajectories to see your course following behaviors and see whether your current cognitive states are in good status or not.'
            +'\n For example, strong course following is correlated with good cognitive states (such as high valid focus/course following or low workload) and weak course following may indicate bad cognitive states. '
            +'\n For gaze/motor simulation (task 2 and task 3), you have steps below. First, consider your past gaze/motor trajectories in the history since current gaze/motor behaviors may be correlated with past gaze/motor behaviors.'
            +'\n Second, consider your past cognitive states trajectories to decide whether your current gaze/motor simulation should follow the current course content pace or not.'
            +'\n For example, good cognitive states (such as high valid focus/course following or low workload) is correlated with strong course following. So your gaze/motor simulation should be consistent with current course contents. '
            +'\n However, bad cognitive states (such as high workload) may make you keep your gaze/motor AOI as previous gaze/motor trajectories or other potential random AOIs. '
            +'\n Finally, note that gaze behaviors do not have to be the same as motor behaviors. Although they are correlated, gaze and motor represent different behaviors (watching AOI and mouse moving AOI) respectively. '
            +'\n For question choice simulation (task 4), you have steps below. First, analyze correlations between past slide AOIs and past gaze/motor trajectories to see your course following behaviors, which are correlated with your question answering performance.'
            +'\n For example, good course following may indicate correct question answers but bad course following may indicate wrong question answers. '
            +'\n Second, consider your past cognitive states trajectories in the history, which may also affect your question answering performance.'
            +'\n For example, good cognitive states may indicate correct question answers but bad cognitive states may indicate wrong question answers. '
            +'\n Third, find related course contents with the question contents and choice contents to get the correct choice. '
            +'\n Finally, according to all information above, estimate whether you could answer each question correct or not. If yes, you give the correct choice. If no, you give the potential wrong choice. '
            +'\n Note that your simulation for all transcripts MUST Not be always the same. It should vary according to course materials and the interaction among different behaviors as depicted above. ' 
        )
        
        simulation_stratey_standard = '\n Your # simulation strategy #: Finish the four tasks according to the instructions.'
        simulation_stratey_cot = '\n Your # simulation strategy #: Instead of directly giving responses of four tasks, you can finish each task one by one. For each task, think of how you can break it into different steps for consideration and then go over all steps to finish your simulation.'
        simulation_stratey_react = '\n Your # simulation strategy #: Think of the order to finish the four tasks. You should also reason and think how different results of different tasks could affect results of other tasks. According to the reasoning, you can also break it into different steps for consideration and then go over all steps to finish your simulation.'
        simulation_stratey_standard_cog = simulation_stratey_standard + '\n\n' + cog_str
        simulation_stratey_cot_cog = simulation_stratey_cot + '\n\n' + cog_str
        simulation_stratey_react_cog = simulation_stratey_react + '\n\n' + cog_str
        
        simulation_stratey_dict = {'standard':simulation_stratey_standard,'cot':simulation_stratey_cot,'react':simulation_stratey_react,'standard_cog':simulation_stratey_standard_cog,'cot_cog':simulation_stratey_cot_cog,'react_cog':simulation_stratey_react_cog}
        
        simulation_stratey = simulation_stratey_dict[sim_strategy]
        
        warning = '\n Please use the # simulation strategy # to finish the four tasks.'


        warning += ('\n Note that: 1. your response for task 1, task 2,task 3 must be answered together in exactly the format below: \n Transcript ID: [value], WORKLOAD: [value], CURIOSITY: [value], VALID FOCUS: [value], COURSE FOLLOW: [value], ENGAGEMENT: [value], CONFUSION: [value], WATCH AOI: [aoi id], MOUSE MOVE AOI: [aoi id].'
                +f'\n You MUST give response value for each task for each transcript from transcript {sentence_id_list[0]} to transcript {sentence_id_list[-1]} in the slide.'
                +'\n You MUST give response value instead of blank for mouse moving simulation even if there is no related history data in mouse moving. '
                +'\n 2. your response for task 4 must be exactly the format below: \n Question ID: [id value], Question Choice: [choice]. '
                +'\n Make sure you have the ID symbol after Question symbol and The [choice] must be only one of A,B,C,D without other words.'
                +f'\n You MUST give response for each question.\n '
        )

        if self.agent_config['memory_source'] == 'sim':
            warning += (
                '\n Your simulation of gaze, motor, cognitive states and question answering MUST be adapted to the student specific # persona # and Try to # diversify # your simulation according to # student personas #..'
                +'\n For example, for question choice task, your goal is NOT to answer it correctly. Instead, you should mimic the specific student persona, gaze/motor and cognitive states to make the choice. For example, good students may make the correct choice while bad students may make the wrong choice. Similar rules apply to other tasks. '   
                +'\n Your gaze DO NOT have to be the same as motor behaviors as well.'
            )

        warning += '\n You just need to give me responses of four tasks in the format above. Do not give any other output nor reasons.'

        observation = (f'\n Now you are on an online course and here is what the teacher is saying: # current course slide ID: {transcript_id}, slide contents: {transcript_material} #. \n')

        aoi_material_choice_str = self._get_aoi_choice_str(aoi_material_table)

        question_str = f'\n Now you have learned knowledge from the # current course slide ID: {transcript_id}#. Now you will take several post-test questions below to evaluate your learning performance:'
        question_id_list = list(question_content_dict.keys())
        question_id_list.sort()
        for question_id in question_id_list:
            question_content = question_content_dict[question_id]
            choice_content = choice_content_dict[question_id]
            question_str += f'\n Question {question_id}: # {question_content} #. You must select one answer from the question choices: # {choice_content} # \n'
       
        content_prompt = task_overview + task_detail + simulation_stratey + retrieved_memory + observation + aoi_material_choice_str + question_str + example_demo_str + warning 

        if self.agent_config['gpt_type'] in [3,4]:
            messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": content_prompt}]
            llm_response = self._response_llm_gpt(messages)
        elif self.agent_config['gpt_type'] == 0:
            messages = sys_prompt + '\n\n' + content_prompt
            llm_response = self._response_llm_gemini(messages)
        elif self.agent_config['gpt_type'] in [1,2]:
            llm_response = self._response_llm_llama(sys_prompt,content_prompt,self.agent_config['gpt_type'])
        else:
            assert 1==0

        match_cognitive_gaze_motor_list = self._find_match_cognitive_gaze_motor(r"Transcript ID: \[?(\d+\.\d+|\d+)\]?, WORKLOAD: \[?(\d+\.\d+|\d+)\]?, CURIOSITY: \[?(\d+\.\d+|\d+)\]?, VALID FOCUS: \[?(\d+\.\d+|\d+)\]?, COURSE FOLLOW: \[?(\d+\.\d+|\d+)\]?, ENGAGEMENT: \[?(\d+\.\d+|\d+)\]?, CONFUSION: \[?(\d+\.\d+|\d+)\]?, WATCH AOI: \[?(\d+\.\d+|\d+)\]?, MOUSE MOVE AOI: \[?(\d+\.\d+|\d+)\]?",llm_response)
        match_choice_list = self._find_match_choice(r"Question ID: \[?(\d+\.\d+|\d+)\]?, Question Choice: \[?([a-zA-Z])\]?",llm_response)

        self._store_log('Repetition: '+str(current_repetition)+'-'*10)
        self._store_log('sys_prompt: '+sys_prompt)
        self._store_log('content_prompt: '+content_prompt)
        self._store_log('\n\n llm response: '+llm_response+'\n\n')

        if len(match_cognitive_gaze_motor_list) == 0 and len(match_choice_list) == 0:
            current_repetition += 1
            print(f"Agent {self.agent_id} in {transcript_id} Function action_gaze_mouse_cog_question_concise returned None. Repetition {current_repetition} of {repeat_threshold}.")

            if current_repetition <= 3: 
                return self.action_gaze_mouse_cog_question_concise(sentence_id_list,example_demo_str,sim_strategy,retrieved_memory,transcript_id,transcript_material,aoi_material_table,question_content_dict,choice_content_dict,repeat_threshold, current_repetition)
            else:
                print(f"Agent {self.agent_id} in {transcript_id} Function action_gaze_mouse_cog_question_concise Reached repetition threshold ({repeat_threshold}). Exiting.")
                return {},{},{},{},{},{},{},{},{},{},{}
        else:
            agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,gaze_aoi_id,gaze_center_tuple,move_aoi_id,move_center_tuple = self._extract_match_cognitive_gaze_motor(match_cognitive_gaze_motor_list,aoi_material_table)
            agent_choice = self._extract_match_choice(match_choice_list)
            
            return gaze_aoi_id, gaze_center_tuple, move_aoi_id, move_center_tuple, agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion, agent_choice


    def reflect_reason(self,memory_stream_str):
        if len(memory_stream_str) == 0 or memory_stream_str == None: return ''
        
        self._store_log('\n Start reflect_reason '+'-'*40+'\n\n')
        sys_prompt = 'Assume you are a ' + self.persona

        task = ('\n I will give you your past experience histories in an online course, potentially including different modalities like course materials, your gaze watching area of interest (AOI), your computer mouse moving AOI to explore contents on course slides, and your cognitive states. ' + 
            ' Your task is to reflect and reason based on the correlation and connections among and across these different modalities. ' +
            'For example, reflect and reason why such course materials lead to your gaze behaviors, your mouse moving behaviors, and your cognitive states. And how these modalities could influence each other. \n')
        
        warning = ('\n You should directly output your reflection and reasoning results within 50 words. Do not output any other information. \n')
        content_prompt = task + memory_stream_str + warning

        if self.agent_config['gpt_type'] in [3,4]:
            messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": content_prompt}]
            llm_response = self._response_llm_gpt(messages)
        elif self.agent_config['gpt_type'] == 0:
            messages = sys_prompt + '\n\n' + content_prompt
            llm_response = self._response_llm_gemini(messages)
        elif self.agent_config['gpt_type'] in [1,2]:
            llm_response = self._response_llm_llama(sys_prompt,content_prompt,self.agent_config['gpt_type'])
        else:
            assert 1==0

        self._store_log('sys_prompt: '+sys_prompt)
        self._store_log('content_prompt: '+content_prompt)
        self._store_log('\n\n llm response: '+llm_response) 

        return llm_response


    def _select_memory_index(self,memory_stream,current_transcript_id):
        # ['no_memory','random_half_plus_recent_one','all_plus_recent_one','only_recent_one']
        if self.agent_config['forget_effect'] == 'no_memory':
            return []
        elif self.agent_config['forget_effect'] == 'all_plus_recent_one':
            return memory_stream
        elif self.agent_config['forget_effect'] == 'random_half_plus_recent_one':
            half_length = len(memory_stream) // 2
            retrieved_memory_stream = random.sample(memory_stream, half_length)
            return retrieved_memory_stream
        elif self.agent_config['forget_effect'] == 'only_recent_one':
            return memory_stream[-1:]
        else:
            assert 1 == 0
        

    def retrieve_memory(self,memory_stream_list,current_transcript_id):
        memory_stream_list_sorted = sorted(memory_stream_list, key=lambda x: x['transcript_id'])
        memory_stream_len = len(memory_stream_list_sorted)

        memory_stream_list_filtered = self._select_memory_index(memory_stream_list_sorted,current_transcript_id)
        # memory_stream_list_filtered = memory_stream_list_sorted
        
        memory_name_dict = {'PM':['gaze_aoi_id'],'MM':['motor_aoi_id'],'CM':['workload','curiosity','valid_focus','course_follow','engagement','confusion']}
        composed_memory_stream = []
        for mi,memory_element in enumerate(memory_stream_list_filtered):
            transcript_id = memory_element['transcript_id']
            transcript = memory_element['observation']
            action_dict = memory_element['action']
            action_name_list = list(action_dict.keys())

            memory_element_retrieve = {'transcript_id':transcript_id}
            if 'reflection' in list(memory_element.keys()) and self.agent_config['reflection_choice'] == 'yes':
                memory_element_retrieve['reflection'] = memory_element['reflection']
            memory_element_retrieve['observation'] = transcript if 'KM' in self.memory_component_choice_list else ''
            memory_element_retrieve['action'] = {}
            for memory_component_name in self.memory_component_choice_list:
                if memory_component_name == 'KM': continue
                for per_metric in memory_name_dict[memory_component_name]:
                    if per_metric not in action_name_list:
                        memory_element_retrieve['action'][per_metric] = None
                    else:
                        memory_element_retrieve['action'][per_metric] = action_dict[per_metric]
            composed_memory_stream.append(memory_element_retrieve)

        return composed_memory_stream

    def summarize_transcripts_llm(self,memory_stream_list):
        if len(memory_stream_list) == 0 or memory_stream_list == None: return ''
        
        self._store_log('\n Start summarize_transcripts '+'-'*40+'\n\n')
        sys_prompt = 'You are an intelligent assistant who is good at summarizing course materials.'

        task = ('\n I will give you a list of course transcripts below. Your task is to summarize these course transcripts within 50 words.\n ')
        
        transcripts_str = ''
        for memory_element in memory_stream_list:
            transcript_id = memory_element['transcript_id']
            transcript_content = memory_element['observation']
            transcripts_str += f'#Transcript: {transcript_id}#: {transcript_content}. \n'

        warning = ('\n You should directly output your summarized transcripts within 50 words. Do not output any other information. \n')
        content_prompt = task + transcripts_str + warning

        if self.agent_config['gpt_type'] in [3,4]:
            messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": content_prompt}]
            llm_response = self._response_llm_gpt(messages)
        elif self.agent_config['gpt_type'] == 0:
            messages = sys_prompt + '\n\n' + content_prompt
            llm_response = self._response_llm_gemini(messages)
        elif self.agent_config['gpt_type'] in [1,2]:
            llm_response = self._response_llm_llama(sys_prompt,content_prompt,self.agent_config['gpt_type'])
        else:
            assert 1==0

        self._store_log('sys_prompt: '+sys_prompt)
        self._store_log('content_prompt: '+content_prompt)
        self._store_log('\n\n llm response: '+llm_response) 

        return llm_response

    def summarize_gaze_llm(self,transcript_list):
        if len(transcript_list) == 0 or transcript_list == None: return ''
        
        self._store_log('\n Start summarize_gaze '+'-'*40+'\n\n')
        sys_prompt = 'You are an intelligent assistant who is good at summarizing course materials and capturing student gaze behaviors.'

        task = ('\n I will give you a list of transcripts of slide areas on the course, which are watched by a student sequentially. ' +
                'Your task is to summarize student gaze behaviors on these course transcripts within 50 words. ')
        transcripts_str = '\n Here are the course areas watched by the student.\n'
        for t_id,transcript_content in enumerate(transcript_list):
            transcripts_str += f'#Gaze on Transcript: {t_id}#: {transcript_content}. \n'
        
        warning = ('\n You should directly output your summary within 50 words. Do not output any other information. \n')
        content_prompt = task + transcripts_str + warning

        if self.agent_config['gpt_type'] in [3,4]:
            messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": content_prompt}]
            llm_response = self._response_llm_gpt(messages)
        elif self.agent_config['gpt_type'] == 0:
            messages = sys_prompt + '\n\n' + content_prompt
            llm_response = self._response_llm_gemini(messages)
        elif self.agent_config['gpt_type'] in [1,2]:
            llm_response = self._response_llm_llama(sys_prompt,content_prompt,self.agent_config['gpt_type'])
        else:
            assert 1==0

        self._store_log('sys_prompt: '+sys_prompt)
        self._store_log('content_prompt: '+content_prompt)
        self._store_log('\n\n llm response: '+llm_response) 

        return llm_response

    def summarize_motor_llm(self,transcript_list):
        if len(transcript_list) == 0 or transcript_list == None: return ''
        
        self._store_log('\n Start summarize_motor '+'-'*40+'\n\n')
        sys_prompt = 'You are an intelligent assistant who is good at summarizing course materials and capturing student mouse moving behaviors in the online course.'

        task = ('\n I will give you a list of transcripts of slides areas on the course, which are explored by a student sequentially using the computer mouse in the online course. ' +
                'Your task is to summarize student mouse moving exploration behaviors on these course transcripts within 50 words. ')
        transcripts_str = '\n Here are the course transcripts explored by the student.\n'
        for t_id,transcript_content in enumerate(transcript_list):
            transcripts_str += f'#Mouse on Transcript: {t_id}#: {transcript_content}. \n'
        
        warning = ('\n You should directly output your summary within 50 words. Do not output any other information. \n')
        content_prompt = task + transcripts_str + warning

        if self.agent_config['gpt_type'] in [3,4]:
            messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": content_prompt}]
            llm_response = self._response_llm_gpt(messages)
        elif self.agent_config['gpt_type'] == 0:
            messages = sys_prompt + '\n\n' + content_prompt
            llm_response = self._response_llm_gemini(messages)
        elif self.agent_config['gpt_type'] in [1,2]:
            llm_response = self._response_llm_llama(sys_prompt,content_prompt,self.agent_config['gpt_type'])
        else:
            assert 1==0

        self._store_log('sys_prompt: '+sys_prompt)
        self._store_log('content_prompt: '+content_prompt)
        self._store_log('\n\n llm response: '+llm_response) 

        return llm_response


    def summarize_transcripts(self,memory_stream_list):
        if len(memory_stream_list) == 0 or memory_stream_list == None: return ''
        transcripts_str = ''
        for memory_element in memory_stream_list:
            transcript_id = memory_element['transcript_id']
            # transcript_content = memory_element['observation']
            transcripts_str += f'#Slide: {transcript_id}#:  \n'
            course_material_item = self.course_dataset_agent[self.course_dataset_agent['slide_id_from_zero']==transcript_id]
            tiny_transcript_id_list = list(set(course_material_item['transcript_id']))
            tiny_transcript_id_list.sort()
            transcripts_str += self._get_transcript_str(tiny_transcript_id_list)
            transcripts_str += '\n'
            # topic_sum = slide_summary_dict['video_'+str(self.video_id)][transcript_id]
            # transcripts_str += f'#Slide: {transcript_id}#: this slide talks about {topic_sum}. \n'
        return transcripts_str

    def summarize_gaze_max(self,gaze_tuple_list):
        if len(gaze_tuple_list) == 0 or gaze_tuple_list == None: return ''
        transcripts_str = '# Gaze Watch AOI #: A summary of your gaze trajectory is below. \n'
        
        valid_num = 0
        for d in gaze_tuple_list:
            if d[0] == None or len(d[0]) == 0: continue 
            aoi_id_dict,transcript_id = d[0],d[1]
            value_counts = Counter(aoi_id_dict.values())
            aoi_id_frequent = max(value_counts, key=value_counts.get)
            if aoi_id_frequent == None: continue
            aoi_material_match = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            aoi_material_match_item = aoi_material_match[aoi_material_match['aoi_id']==aoi_id_frequent]
            aoi_content = aoi_material_match_item['aoi_content'].values[0]
            transcripts_str += f'During Slide: {transcript_id}, the student gaze focused more on the content below: # {aoi_content} #. \n'
            valid_num += 1
        
        if valid_num == 0: return ''
        
        return transcripts_str

    def summarize_gaze(self,gaze_tuple_list):
        if len(gaze_tuple_list) == 0 or gaze_tuple_list == None: return ''
        transcripts_str = '# Gaze Watch AOI #: A summary of your gaze trajectory is below. \n'
        valid_num = 0
        for d in gaze_tuple_list:
            if d[0] in [-1,None] or len(d[0]) == 0: continue 
            aoi_id_dict,transcript_id = d[0],d[1]
            # aoi_material_match = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            aoi_id_dict_sorted = dict(sorted(aoi_id_dict.items()))
            action_trajectory = list(aoi_id_dict_sorted.values())
            
            # for aoi_id_each in action_trajectory:
            action_trajectory = [str(a_t) if a_t is not None else ' ' for a_t in action_trajectory]
            action_trajectory = ','.join(action_trajectory)
            # aoi_material_match_item = aoi_material_match[aoi_material_match['aoi_id']==aoaoi_id_eachi_id]
            # aoi_content = aoi_material_match_item['aoi_content'].values[0]
            # gaze_aoi_trajectory = 
            transcripts_str += f'During Slide: {transcript_id}, gaze AOI ID trajectory is: {action_trajectory}\n'
            valid_num += 1
        if valid_num == 0: return ''
        
        return transcripts_str

    def summarize_motor_max(self,motor_tuple_list):
        if len(motor_tuple_list) == 0 or motor_tuple_list == None: return ''
        transcripts_str = '# Mouse Move AOI #: A summary of your mouse moving trajectory is below. \n'
        # print('motor_tuple_list: ',motor_tuple_list)
        valid_num = 0
        for d in motor_tuple_list:
            if d[0] == None or len(d[0]) == 0: continue 
            aoi_id_dict,transcript_id = d[0],d[1]
            value_counts = Counter(aoi_id_dict.values())
            aoi_id_frequent = max(value_counts, key=value_counts.get)
            # print('transcript_id: ',transcript_id)
            # print('aoi_id_frequent: ',aoi_id_frequent)
            if aoi_id_frequent == None: continue
            aoi_material_match = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            # print('aoi_material_match: ',aoi_material_match)
            aoi_material_match_item = aoi_material_match[aoi_material_match['aoi_id']==aoi_id_frequent]
            aoi_content = aoi_material_match_item['aoi_content'].values[0]
            transcripts_str += f'During Slide: {transcript_id}, the student moved mouse more on the content below: # {aoi_content} #. \n'
            valid_num += 1
        
        if valid_num == 0: return ''

        return transcripts_str


    def summarize_motor(self,motor_tuple_list):
        if len(motor_tuple_list) == 0 or motor_tuple_list == None: return ''
        transcripts_str = '# Mouse Move AOI #: A summary of your mouse moving trajectory is below. \n'
        valid_num = 0
        for d in motor_tuple_list:
            if d[0] in [-1,None] or len(d[0]) == 0: continue 
            aoi_id_dict,transcript_id = d[0],d[1]
            # aoi_material_match = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            aoi_id_dict_sorted = dict(sorted(aoi_id_dict.items()))
            action_trajectory = list(aoi_id_dict_sorted.values())
            
            # for aoi_id_each in action_trajectory:
            action_trajectory = [str(a_t) if a_t is not None else ' ' for a_t in action_trajectory]
            action_trajectory = ','.join(action_trajectory)
            # aoi_material_match_item = aoi_material_match[aoi_material_match['aoi_id']==aoaoi_id_eachi_id]
            # aoi_content = aoi_material_match_item['aoi_content'].values[0]
            # gaze_aoi_trajectory = 
            transcripts_str += f'During Slide: {transcript_id}, mouse moving AOI ID trajectory is: {action_trajectory}\n'
            valid_num += 1
        
        if valid_num == 0: return ''
        return transcripts_str

    def summarize_actions(self,memory_stream_list):
        if len(memory_stream_list) == 0 or memory_stream_list == None: return ''

        transcript_id_start = memory_stream_list[0]['transcript_id']
        transcript_id_end = memory_stream_list[-1]['transcript_id']
        summarized_actions_str = f'Here are your experience trajectory from Slide {transcript_id_start} to Slide {transcript_id_end}. \n'
        summarized_actions_dict = {}
        for memory_element in memory_stream_list:
            transcript_id = memory_element['transcript_id']
            action_dict = memory_element['action']
            if len(action_dict) == 0: continue
            action_list = list(action_dict.keys())
            
            for action_name in action_list:
                action_value = memory_element['action'][action_name]
                if action_name in ['gaze_aoi_id','motor_aoi_id']:    
                    if action_name not in list(summarized_actions_dict.keys()):
                        summarized_actions_dict[action_name] = [(action_value,transcript_id)]
                    else:
                        summarized_actions_dict[action_name].append((action_value,transcript_id))
                        
                elif action_name in ['workload','curiosity','valid_focus','course_follow','engagement','confusion']:    
                    if action_value == None or len(action_value) == 0: 
                        action_trajectory = ''
                    else:
                        action_value_sorted = dict(sorted(action_value.items()))
                        action_trajectory = list(action_value_sorted.values())
                        action_trajectory = [str(round(a_t,2)) if a_t is not None else ' ' for a_t in action_trajectory]
                        action_trajectory = ','.join(action_trajectory)
                    if action_name not in list(summarized_actions_dict.keys()):
                        summarized_actions_dict[action_name] = f'Your # {action_name} # trajectory is: {action_trajectory},'
                    else:
                        summarized_actions_dict[action_name] = summarized_actions_dict[action_name] + action_trajectory + ', '

        action_list_sum = list(summarized_actions_dict.keys())

        for action in action_list_sum:
            if action not in ['gaze_aoi_id','motor_aoi_id']:
                summarized_actions_str += summarized_actions_dict[action] + '.\n'

        if 'gaze_aoi_id' in action_list_sum and len(summarized_actions_dict['gaze_aoi_id']) != 0 and summarized_actions_dict['gaze_aoi_id'] != None:
            sum_gaze_content = self.summarize_gaze(summarized_actions_dict['gaze_aoi_id'])
            summarized_actions_str += sum_gaze_content

        if 'motor_aoi_id' in action_list_sum and len(summarized_actions_dict['motor_aoi_id']) != 0 and summarized_actions_dict['motor_aoi_id'] != None:
            sum_motor_content = self.summarize_motor(summarized_actions_dict['motor_aoi_id'])
            summarized_actions_str += sum_motor_content
        
        return summarized_actions_str
   
    def summarize_aois(self,memory_stream_list):
        if len(memory_stream_list) == 0 or memory_stream_list == None: return ''
        transcripts_str = ''
        for memory_element in memory_stream_list:
            transcript_id = memory_element['transcript_id']
            transcripts_str += f'#Slide: {transcript_id}#: \n'
            aoi_material_table = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            transcripts_str += self._get_aoi_choice_str(aoi_material_table)
            transcripts_str += '\n'
        return transcripts_str

    def summarize_memory(self,memory_stream_list):
        transcript_id_start = memory_stream_list[0]['transcript_id']
        transcript_id_end = memory_stream_list[-1]['transcript_id']
        summarized_transcripts = '# ' + self.summarize_transcripts(memory_stream_list) + ' #.'
        summarized_actions = self.summarize_actions(memory_stream_list)
        summarized_aois = self.summarize_aois(memory_stream_list)
        memory_string = (f'\n # Here are old histories from slide {transcript_id_start} to slide {transcript_id_end} and your corresponding experience. #\n' + 
                        'The contents of the past slides are listed below: \n' + summarized_transcripts + '.\n\n' + 
                        'The contents of AOIs on the past slides are listed below: \n' + summarized_aois + '.\n\n' + 
                        'During these course slides, your actions and states are listed below: \n' + summarized_actions + '.\n\n')
        
        return memory_string

    def _generate_memory_string(self,memory_stream):
        # Warning: revising this function will affect both memory stream to agents as well as reflection & reasoning process
        if memory_stream == None or len(memory_stream) == 0:
            return '\n\n There is no memory stream or history records for you.\n', ''
        memory_string = '\n\n You have experienced such histories in the course depicted below. \n'
        
        summarized_memories = self.summarize_memory(memory_stream)
        memory_string += summarized_memories

        if 'reflection' in list(memory_stream[-1].keys()):
            reflection_words = memory_stream[-1]['reflection']
            if len(reflection_words) != 0: 
                reflect_string = f'\n Your # self-reflection and self-reasoning # for previous course histories and experience is below: {reflection_words}. \n'
            else:
                reflect_string = ''
        else:
            reflect_string = ''

        return memory_string, reflect_string



    def _generate_memory_string_old(self,memory_stream,memory_hold_threshold = 10):
        # Warning: revising this function will affect both memory stream to agents as well as reflection & reasoning process
        if memory_stream == None or len(memory_stream) == 0:
            return 'There is no memory stream or history records for you.\n', ''
        memory_string = 'You have experienced such histories in the course depicted below. \n'
        
        if len(memory_stream) > memory_hold_threshold:
            memory_stream_exact = memory_stream[-memory_hold_threshold:]
            summarized_memories = self.summarize_memory(memory_stream[:-memory_hold_threshold])
            memory_string += summarized_memories
        else:
            memory_stream_exact = memory_stream

        memory_string += f'\n # Here are the most recent histories of transcripts and your experience. #\n'
        for memory_element in memory_stream_exact:
            transcript = memory_element['observation']
            if len(transcript)==0: continue
            transcript_id = memory_element['transcript_id']
            action_dict = memory_element['action']
            memory_string += f'\n # Course Transcript {transcript_id}#: # {transcript} #\n'
            if len(action_dict) == 0: continue
            memory_string += f'\n Your actions for the # Transcript {transcript_id}# is below:\n'
            action_list = list(action_dict.keys())

            aoi_material_match = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            for action_name in action_list:
                action_value = memory_element['action'][action_name]
                if action_value == None:
                    continue
                
                if action_name == 'gaze_aoi_id':    
                    if action_value == -1:    
                        memory_string += f'# Gaze Watch AOI #: You were not watching on any valid AOIs on slides. \n'
                    else:
                        aoi_material_match_item = aoi_material_match[aoi_material_match['aoi_id']==action_value]
                        aoi_content = aoi_material_match_item['aoi_content'].values[0]
                        memory_string += f'# Gaze Watch AOI #: You were watching one AOI (Area of Interest) on slides whose contents are: # {aoi_content} #, \n'
                elif action_name == 'motor_aoi_id':   
                    if action_value == -1:     
                        memory_string += f'# Mouse Move AOI #: You were not moving the mouse on any valid AOIs on slides. \n'
                    else:
                        aoi_material_match_item = aoi_material_match[aoi_material_match['aoi_id']==action_value]
                        aoi_content = aoi_material_match_item['aoi_content'].values[0]             
                        memory_string += f'# Mouse Move AOI #: You were moving the mouse to explore contents on one AOI (Area of Interest) on slides whose contents are: # {aoi_content} #, \n'
                elif action_name == 'workload':   
                    action_value = round(action_value,2)  
                    memory_string += f'# WORKLOAD #: Your workload was # {action_value} #, \n'
                elif action_name == 'curiosity':  
                    action_value = round(action_value,2)   
                    memory_string += f'# CURIOSITY #: Your curiosity to explore course contents was # {action_value} #, \n'
                elif action_name == 'valid_focus':
                    action_value = round(action_value,2)
                    memory_string += f'# VALID FOCUS #: Your valid focus extent was # {action_value} #, \n'
                elif action_name == 'course_follow':
                    action_value = round(action_value,2)
                    memory_string += f'# COURSE FOLLOW #: Your course following extent was # {action_value} #, \n'
                elif action_name == 'engagement':
                    action_value = round(action_value,2)
                    memory_string += f'# ENGAGEMENT #: Your engagement value was # {action_value} #, \n'
                elif action_name == 'confusion':
                    action_value = round(action_value,2)
                    memory_string += f'# CONFUSION #: Your confusion value was # {action_value} #, \n'

        cog_metric_str = ('\n\n Note that WORKLOAD is a float numeric value from 0 (very low) to 1 (very high) to indicate your mental workload in the course. ' + 
            'CURIOSITY is a float numeric value from 0 (very low) to 1 (very high) to indicate to what extent you are curious to explore the course contents. ' + 
            'VALID FOCUS is a float numeric value from 0 (very low) to 1 (very high) to indicate to what extent you could focus on the valid contents in the course. ' + 
            'COURSE FOLLOW is a float numeric value from 0 (very low) to 1 (very high) to indicate to what extent you could follow the course pace. ' + 
            'ENGAGEMENT is an integer value either 0 (not engaged) or 1 (engaged) to indicate whether you could engage in the course or not. ' + 
            'CONFUSION is an integer value either 0 (not confused) or 1 (confused) to indicate whether you feel confused about course contents or not. \n\n' )

        memory_string = memory_string + cog_metric_str

        if 'reflection' in list(memory_stream[-1].keys()):
            reflection_words = memory_stream[-1]['reflection']
            if len(reflection_words) != 0: 
                reflect_string = f'\n Your # self-reflection and self-reasoning # for previous course histories and experience is below: {reflection_words}. \n'
            else:
                reflect_string = ''
        else:
            reflect_string = ''

        return memory_string, reflect_string

    def load_memory_stream(self,current_transcript_id):
        
        if self.agent_config['memory_source'] == 'real':
            with open(self.user_memory_file, 'r') as json_file:
                self.agent_real_memory_stream = json.load(json_file)
            retrieved_memory_stream = self.retrieve_memory(self.agent_real_memory_stream,current_transcript_id)
            
            return retrieved_memory_stream
        elif self.agent_config['memory_source'] == 'sim':
            with open(self.agent_memory_file, 'r') as json_file:
                self.agent_sim_memory_stream = json.load(json_file)
            retrieved_memory_stream = self.retrieve_memory(self.agent_sim_memory_stream,current_transcript_id)
            
            return retrieved_memory_stream

    def add_to_agent_memory(self,memory_element):
        self._store_log('\n\n Adding new agent memory \n\n')
        with open(self.agent_memory_file, 'r') as json_file:
            self.agent_sim_memory_stream = json.load(json_file)
        
        self.agent_sim_memory_stream.append(memory_element)

        with open(self.agent_memory_file, 'w') as json_file:
            json.dump(self.agent_sim_memory_stream, json_file, indent=4) 

    def add_to_user_memory(self,memory_element):
        self._store_log('\n\n Adding new user memory \n\n')
        with open(self.user_memory_file, 'r') as json_file:
            self.agent_real_memory_stream = json.load(json_file)
        
        self.agent_real_memory_stream.append(memory_element)

        with open(self.user_memory_file, 'w') as json_file:
            json.dump(self.agent_real_memory_stream, json_file, indent=4) 

    def _response_llm_llama(self,sys_prompt,content_prompt,model_type,timeout=120):
        assert model_type in [1,2]
        except_waiting_time = 1
        max_waiting_time = 32
        current_sleep_time = 0.5

        if model_type == 1:
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        else:
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
        headers = {"Authorization": f"Bearer hf_xxxxxxxxxxxxxx",
            "Content-Type": "application/json",}

        json_body = {
            "inputs": f"[INST] <<SYS>> {sys_prompt}. <<SYS>> {content_prompt} [/INST] ",
                "parameters": {"temperature":0.001,'max_new_tokens':2048}
            }
        data = json.dumps(json_body)

        start_time = time.time()   
        response = ''
        while response == '':
            try:
                response = requests.request("POST", API_URL, headers=headers, data=data)
            except Exception as e:
                print(self.agent_id,e)
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time-1)
        
        end_time = time.time()   
        print(self.agent_id,'llm response time: ',end_time-start_time)  
        response_output = json.loads(response.content.decode("utf-8"))
        try:
            response_str = response_output[0]['generated_text'] 
        except:
            response_str = ''
        return response_str

    def _response_llm_gemini(self,message_list,timeout=120):
        except_waiting_time = 1
        max_waiting_time = 32
        current_sleep_time = 0.5
        max_num = 3
        current_num = 0

        genai.configure(api_key='')
        # temperature Values can range from [0.0,1.0], inclusive. A value closer to 1.0 will produce responses that are more varied and creative, while a value closer to 0.0 will typically result in more straightforward responses from the model
        google_model = genai.GenerativeModel('gemini-pro')

        start_time = time.time()
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            # stop_sequences=['x'],
            # max_output_tokens=1000,
            temperature=0
        )
        response = ''
        while response == '':
            
            try:
                response = google_model.generate_content(message_list,generation_config=generation_config)
                response = str(response.text)
            except Exception as e:
                print(self.agent_id,e)
                print(response)
                response = ''
                
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time-1)

            current_num += 1
            if current_num > max_num:
                break
        
        end_time = time.time()   
        print(self.agent_id,'llm response time: ',end_time-start_time)     
        return response
 

    def _response_llm_gpt(self,message_list,timeout=120):
        response = ''
        except_waiting_time = 1
        max_waiting_time = 32
        current_sleep_time = 0.5
        if self.agent_config['gpt_type'] == 3:
            model_name = "gpt-3.5-turbo-1106" 
        elif self.agent_config['gpt_type'] == 4: 
            model_name = 'gpt-4-1106-preview'
        else:
            print('GPT model name error.')
            assert 1==0
        start_time = time.time()
        while response == '':
            try:
                completion = client.chat.completions.create(
                    model=model_name, # gpt-4-1106-preview, gpt-4
                    messages=message_list,
                    temperature=0,
                    timeout = timeout,
                    max_tokens=3000
                    )
                
                response = completion.choices[0].message.content
            except Exception as e:
                print(self.agent_id,e)
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time-1)
        end_time = time.time()   
        print(self.agent_id,'llm response time: ',end_time-start_time)     
        return response
  
    def _get_real_gaze(self,during_table):
        if len(during_table) == 0: return None, (None, None)
        user_aoi_id = during_table['gaze_aoi_id'].mode().values[0]
        aoi_table = during_table[during_table['gaze_aoi_id']==user_aoi_id]
        user_aoi_center_tuple = aoi_table['gaze_aoi_center_x_ratio'].values[0],aoi_table['gaze_aoi_center_y_ratio'].values[0]
        if user_aoi_id == -1: return None, (None, None)
        return user_aoi_id, user_aoi_center_tuple

    def _get_real_motor(self,during_table):
        if len(during_table) == 0: return None, (None, None)
        user_aoi_id = during_table['mouse_aoi_id'].mode().values[0]
        aoi_table = during_table[during_table['mouse_aoi_id']==user_aoi_id]
        user_aoi_center_tuple = aoi_table['mouse_aoi_center_x_ratio'].values[0],aoi_table['mouse_aoi_center_y_ratio'].values[0]
        if user_aoi_id == -1: return None, (None, None)
        return user_aoi_id, user_aoi_center_tuple

    def _get_real_cognitive_state(self,during_table):
        if len(during_table) == 0: return None, None, None, None, None, None 
        stationary_entropy_valid_table = during_table[during_table['gaze_entropy_stationary_norm']!=-1]
        transition_entropy_valid_table = during_table[during_table['gaze_entropy_transition_norm']!=-1]
        user_workload = None if len(stationary_entropy_valid_table) == 0 else round(stationary_entropy_valid_table['gaze_entropy_stationary_norm'].mean(),2)
        user_curiosity = None if len(transition_entropy_valid_table) == 0 else round(transition_entropy_valid_table['gaze_entropy_transition_norm'].mean(),2)

        user_valid_focus = round(during_table['valid_focus'].mean(),2) # average focus extent (0,1)
        user_course_follow = round(during_table['course_follow'].mean(),2) # average course follow extent (0,1)
        user_engagement = round(during_table['engagement'].mean(),2) # average course follow extent (0,1)
        user_confusion = round(during_table['confusion'].mean(),2) # average course follow extent (0,1)
        # user_engagement = 0 if during_table['engagement'].mean() != 1 else 1 # once face-lost happens, engagement = 0. Otherwise, engagement = 1.
        # user_confusion = 1 if during_table['confusion'].mean() != 0 else 0 # once click report confusion, confusion = 1. Otherwise, confusion = 0.

        return user_workload, user_curiosity, user_valid_focus, user_course_follow, user_engagement, user_confusion 

    def _simulate_gaze_motor_cog_question(self,example_demo_str,sim_strategy,retrieved_memory,transcript_id,transcript_material,aoi_material_item,user_table,question_item,user_answer_item):
        question_id_list = list(set(question_item['question_id']))
        question_id_list.sort()
        question_content_dict, choice_content_dict, correct_answer, user_answer = {}, {}, {}, {}
        for question_id in question_id_list:
            question_item_per = question_item[question_item['question_id']==question_id]
            question_content_dict[question_id] = question_item_per['question_content'].values[0]
            choice_content_dict[question_id] = question_item_per['choice_content'].values[0]
            correct_answer[question_id] = self.choice_map[question_item_per['correct_answer'].values[0]]

            user_answer_item_per = user_answer_item[user_answer_item['question_id']=='test_q'+str(question_id)]

            if len(user_answer_item_per) == 0 or user_answer_item_per['choice'].values[0] is None or user_answer_item_per is None or user_answer_item_per['question_id'] is None:
                user_answer[question_id] = None 
            else:
                try:
                    user_answer[question_id] = self.choice_map[user_answer_item_per['choice'].values[0]]
                except:
                    user_answer[question_id] = None

        sentence_id_list = list(set(aoi_material_item['transcript_id']))
        sentence_id_list.sort()

        agent_gaze_aoi_id, agent_gaze_aoi_center_tuple, agent_motor_aoi_id, agent_motor_aoi_center_tuple, agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,agent_answer = self.action_gaze_mouse_cog_question_concise(sentence_id_list,example_demo_str,sim_strategy,retrieved_memory,transcript_id,transcript_material,aoi_material_item,question_content_dict,choice_content_dict) 
        
        gaze_aoi_accuracy, gaze_aoi_distance, motor_aoi_accuracy, motor_aoi_distance, workload_diff, curiosity_diff, valid_focus_diff, follow_ratio_diff, engagement_accuracy, confusion_accuracy = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        user_gaze_aoi_id,user_gaze_aoi_center_tuple,user_motor_aoi_id,user_motor_aoi_center_tuple = {}, {}, {}, {}
        user_workload,user_curiosity,user_valid_focus,user_course_follow,user_engagement,user_confusion = {}, {}, {}, {}, {}, {}
        choice_similarity_item,accuracy_similarity_item = {},{}
        for sentence_id in sentence_id_list:
            user_table_per = user_table[user_table['transcript_id']==sentence_id]
            user_gaze_table = user_table_per[user_table_per['gaze_aoi_id']!=-1]
            user_motor_table = user_table_per[user_table_per['mouse_aoi_id']!=-1]

            if agent_gaze_aoi_id == None or len(agent_gaze_aoi_id) == 0 or len(user_gaze_table) == 0:
                gaze_aoi_accuracy[sentence_id] = None
                gaze_aoi_distance[sentence_id] = None
                user_gaze_aoi_id[sentence_id] = None
                user_gaze_aoi_center_tuple[sentence_id] = (None,None)
                if self.agent_config['memory_source'] == 'real':
                    agent_gaze_aoi_id[sentence_id] = None
                    agent_gaze_aoi_center_tuple[sentence_id] = (None,None)

                if len(agent_gaze_aoi_id) == 0 or sentence_id not in list(agent_gaze_aoi_id.keys()):
                    agent_gaze_aoi_id[sentence_id] = None
                    agent_gaze_aoi_center_tuple[sentence_id] = (None,None)
                
            else:
                user_gaze_aoi_id_piece, user_gaze_aoi_center_tuple_piece = self._get_real_gaze(user_gaze_table)
                user_gaze_aoi_id[sentence_id] = user_gaze_aoi_id_piece
                user_gaze_aoi_center_tuple[sentence_id] = user_gaze_aoi_center_tuple_piece

                if sentence_id in list(agent_gaze_aoi_id.keys()):
                    gaze_aoi_accuracy[sentence_id] = 1 if agent_gaze_aoi_id[sentence_id] == user_gaze_aoi_id[sentence_id] else 0
                else:
                    agent_gaze_aoi_id[sentence_id] = None
                    gaze_aoi_accuracy[sentence_id] = None 
                if sentence_id in list(agent_gaze_aoi_center_tuple.keys()):
                    gaze_aoi_distance[sentence_id] = self._calculate_distance(agent_gaze_aoi_center_tuple[sentence_id],user_gaze_aoi_center_tuple[sentence_id])
                else:
                    agent_gaze_aoi_center_tuple[sentence_id] = (None,None)
                    gaze_aoi_distance[sentence_id] = None 

            if agent_motor_aoi_id == None or len(agent_motor_aoi_id) == 0 or len(user_motor_table) == 0:
                motor_aoi_accuracy[sentence_id] = None
                motor_aoi_distance[sentence_id] = None
                user_motor_aoi_id[sentence_id] = None
                user_motor_aoi_center_tuple[sentence_id] = (None,None)
                if self.agent_config['memory_source'] == 'real':
                    agent_motor_aoi_id[sentence_id] = None
                    agent_motor_aoi_center_tuple[sentence_id] = (None,None)
                if len(agent_motor_aoi_id) == 0 or sentence_id not in list(agent_motor_aoi_id.keys()):
                    agent_motor_aoi_id[sentence_id] = None
                    agent_motor_aoi_center_tuple[sentence_id] = (None,None)
                
            else:
                user_motor_aoi_id_piece, user_motor_aoi_center_tuple_piece = self._get_real_motor(user_motor_table)
                user_motor_aoi_id[sentence_id] = user_motor_aoi_id_piece
                user_motor_aoi_center_tuple[sentence_id] = user_motor_aoi_center_tuple_piece

                if sentence_id in list(agent_motor_aoi_id.keys()):
                    motor_aoi_accuracy[sentence_id] = 1 if agent_motor_aoi_id[sentence_id] == user_motor_aoi_id[sentence_id] else 0
                else:
                    agent_motor_aoi_id[sentence_id] = None
                    motor_aoi_accuracy[sentence_id] = None 
                if sentence_id in list(agent_motor_aoi_center_tuple.keys()):
                    motor_aoi_distance[sentence_id] = self._calculate_distance(agent_motor_aoi_center_tuple[sentence_id],user_motor_aoi_center_tuple[sentence_id])
                else:
                    agent_motor_aoi_center_tuple[sentence_id] = (None,None)
                    motor_aoi_distance[sentence_id] = None 

            user_workload_piece,user_curiosity_piece,user_valid_focus_piece,user_course_follow_piece,user_engagement_piece,user_confusion_piece = self._get_real_cognitive_state(user_table_per)
            user_workload[sentence_id] = user_workload_piece
            user_curiosity[sentence_id] = user_curiosity_piece
            user_valid_focus[sentence_id] = user_valid_focus_piece
            user_course_follow[sentence_id] = user_course_follow_piece
            user_engagement[sentence_id] = user_engagement_piece
            user_confusion[sentence_id] = user_confusion_piece
            
            if sentence_id in list(agent_workload.keys()):
                workload_diff[sentence_id] = self._calculate_difference(user_workload[sentence_id],agent_workload[sentence_id]) 
            else:
                agent_workload[sentence_id] = None
                workload_diff[sentence_id] = None 
            if sentence_id in list(agent_curiosity.keys()):
                curiosity_diff[sentence_id] = self._calculate_difference(user_curiosity[sentence_id],agent_curiosity[sentence_id]) 
            else:
                agent_curiosity[sentence_id] = None
                curiosity_diff[sentence_id] = None 
            if sentence_id in list(agent_valid_focus.keys()):
                valid_focus_diff[sentence_id] = self._calculate_difference(user_valid_focus[sentence_id],agent_valid_focus[sentence_id]) 
            else:
                agent_valid_focus[sentence_id] = None
                valid_focus_diff[sentence_id] = None 
            if sentence_id in list(agent_course_follow.keys()):
                follow_ratio_diff[sentence_id] = self._calculate_difference(user_course_follow[sentence_id],agent_course_follow[sentence_id]) 
            else:
                agent_course_follow[sentence_id] = None
                follow_ratio_diff[sentence_id] = None 
            if sentence_id in list(agent_engagement.keys()):
                engagement_accuracy[sentence_id] = self._calculate_difference(user_engagement[sentence_id],agent_engagement[sentence_id]) 
            else:
                agent_engagement[sentence_id] = None
                engagement_accuracy[sentence_id] = None 
            if sentence_id in list(agent_confusion.keys()):
                confusion_accuracy[sentence_id] = self._calculate_difference(user_confusion[sentence_id],agent_confusion[sentence_id])
            else:
                agent_confusion[sentence_id] = None
                confusion_accuracy[sentence_id] = None 

        for question_id in question_id_list:
            if question_id in list(agent_answer.keys()):
                user_accuracy_item = 1 if user_answer[question_id] == correct_answer[question_id] else 0 
                agent_accuracy_item = 1 if agent_answer[question_id] == correct_answer[question_id] else 0 
                choice_similarity_item[question_id] = 1 if user_answer[question_id] == agent_answer[question_id] else 0
                accuracy_similarity_item[question_id] = 1 if user_accuracy_item == agent_accuracy_item else 0
            else:
                agent_answer[question_id] = None 
                choice_similarity_item[question_id] = None
                accuracy_similarity_item[question_id] = None

        return gaze_aoi_accuracy,gaze_aoi_distance,user_gaze_aoi_id,user_gaze_aoi_center_tuple,agent_gaze_aoi_id,agent_gaze_aoi_center_tuple,motor_aoi_accuracy,motor_aoi_distance,user_motor_aoi_id,user_motor_aoi_center_tuple,agent_motor_aoi_id,agent_motor_aoi_center_tuple,workload_diff,curiosity_diff,valid_focus_diff,follow_ratio_diff,engagement_accuracy,confusion_accuracy,user_workload,user_curiosity,user_valid_focus,user_course_follow,user_engagement,user_confusion,agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,choice_similarity_item,accuracy_similarity_item,user_answer,agent_answer,correct_answer


    def _get_transcript_str(self,tiny_transcript_id_list):
        transcript_material = ''
        for tiny_transcript_id in tiny_transcript_id_list: 
            tiny_content = self.transcript_dict_agent[int(tiny_transcript_id)]['content']
            transcript_material += (f'\n # Transcript ID: {tiny_transcript_id} #: content: # {tiny_content} #.')
        return transcript_material

    def agent_run_during_class(self):
        print('self.exist_simulation_transcript_id_list: ',self.exist_simulation_transcript_id_list)
        for transcript_id in self.transcript_id_list_simulation:
            if len(self.exist_simulation_transcript_id_list) != 0:
                if transcript_id in self.exist_simulation_transcript_id_list: continue
                self.exist_simulation_transcript_id_list.sort()
                if transcript_id != (self.exist_simulation_transcript_id_list[-1]+1): continue
           
            print(f'similating user {self.agent_id} in slide: {transcript_id}')
            self._store_log('\n Start Simulation for Slide '+str(transcript_id)+'-'*20+'\n\n')
            # transcript_material = self.transcript_dict_agent[transcript_id]['content']
            course_material_item = self.course_dataset_agent[self.course_dataset_agent['slide_id_from_zero']==transcript_id]
            tiny_transcript_id_list = list(set(course_material_item['transcript_id']))
            tiny_transcript_id_list.sort()
            transcript_material = self._get_transcript_str(tiny_transcript_id_list)
            
            aoi_material_item = self.aoi_material_dataset_agent[self.aoi_material_dataset_agent['slide_id_from_zero']==transcript_id]
            during_item = self.during_dataset_agent[self.during_dataset_agent['slide_id_from_zero']==transcript_id]

            # question_id = question_id_map_slide_dict['video_'+str(self.video_id)][transcript_id][0]
            # question_item = self.question_dataset_agent[self.question_dataset_agent['question_id']==question_id]
            # user_answer_item = self.student_answer_item_dataset_agent[self.student_answer_item_dataset_agent['question_id']=='test_q'+str(question_id)]

            question_id_list = question_id_map_slide_dict['video_'+str(self.video_id)][transcript_id]
            question_id_ext_list = ['test_q' + str(q_id) for q_id in question_id_list]
            question_item = self.question_dataset_agent[self.question_dataset_agent['question_id'].isin(question_id_list)]
            user_answer_item = self.student_answer_item_dataset_agent[self.student_answer_item_dataset_agent['question_id'].isin(question_id_ext_list)]

            retrieved_memory_stream = self.load_memory_stream(transcript_id)
            retrieved_memory_stream_no_reflect_str,reflect_string = self._generate_memory_string(retrieved_memory_stream)
            retrieved_memory_stream_str = retrieved_memory_stream_no_reflect_str + reflect_string

            if self.agent_config['example_demo'] == 'yes':
                example_demo_str = self.obtain_example_demo_str(transcript_id,question_id_list)
            else:
                example_demo_str = ''

            # gaze_aoi_accuracy,gaze_aoi_distance,user_gaze_aoi_id,user_gaze_aoi_center_tuple,agent_gaze_aoi_id,agent_gaze_aoi_center_tuple = self._simulate_gaze(retrieved_memory_stream_str,transcript_id,transcript_material,aoi_material_item,during_item)
            # motor_aoi_accuracy,motor_aoi_distance,user_motor_aoi_id,user_motor_aoi_center_tuple,agent_motor_aoi_id,agent_motor_aoi_center_tuple = self._simulate_motor(retrieved_memory_stream_str,transcript_id,transcript_material,aoi_material_item,during_item)
            # workload_diff,curiosity_diff,valid_focus_diff,follow_ratio_diff,engagement_accuracy,confusion_accuracy,user_workload,user_curiosity,user_valid_focus,user_course_follow,user_engagement,user_confusion,agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion = self._simulate_cognitive_state(retrieved_memory_stream_str,transcript_id,transcript_material,aoi_material_item,during_item)
            sim_strategy = self.agent_config['sim_strategy']
            
            gaze_aoi_accuracy,gaze_aoi_distance,user_gaze_aoi_id,user_gaze_aoi_center_tuple,agent_gaze_aoi_id,agent_gaze_aoi_center_tuple,motor_aoi_accuracy,motor_aoi_distance,user_motor_aoi_id,user_motor_aoi_center_tuple,agent_motor_aoi_id,agent_motor_aoi_center_tuple,workload_diff,curiosity_diff,valid_focus_diff,follow_ratio_diff,engagement_accuracy,confusion_accuracy,user_workload,user_curiosity,user_valid_focus,user_course_follow,user_engagement,user_confusion,agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion,choice_similarity_item,accuracy_similarity_item,user_answer,agent_answer,correct_answer = self._simulate_gaze_motor_cog_question(example_demo_str,sim_strategy,retrieved_memory_stream_str,transcript_id,transcript_material,aoi_material_item,during_item,question_item,user_answer_item) 

            sentence_id_list = list(set(aoi_material_item['transcript_id']))
            sentence_id_list.sort()

            
            if self.agent_config['memory_source'] == 'real':
                user_action_dict = {'gaze_aoi_id': user_gaze_aoi_id, 'motor_aoi_id': user_motor_aoi_id, 'workload': user_workload,'curiosity': user_curiosity,'valid_focus': user_valid_focus,'course_follow': user_course_follow,'engagement': user_engagement,'confusion': user_confusion}
                memory_element = {'transcript_id':transcript_id,'observation':transcript_material,'action':user_action_dict}
                if self.agent_config['reflection_choice'] == 'yes':
                    memory_element['reflection'] = self.reflect_reason(retrieved_memory_stream_no_reflect_str)
                self.add_to_user_memory(memory_element)
            if self.agent_config['memory_source'] == 'sim':
                agent_action_dict = {'gaze_aoi_id': agent_gaze_aoi_id, 'motor_aoi_id': agent_motor_aoi_id, 'workload': agent_workload, 'curiosity': agent_curiosity, 'valid_focus': agent_valid_focus, 'course_follow': agent_course_follow, 'engagement': agent_engagement, 'confusion': agent_confusion}
                memory_element = {'transcript_id':transcript_id,'observation':transcript_material,'action':agent_action_dict}
                if self.agent_config['reflection_choice'] == 'yes':
                    memory_element['reflection'] = self.reflect_reason(retrieved_memory_stream_no_reflect_str)
                self.add_to_agent_memory(memory_element)

            for sentence_id in sentence_id_list:
                user_dur_result_list = [user_gaze_aoi_id[sentence_id],user_gaze_aoi_center_tuple[sentence_id][0],user_gaze_aoi_center_tuple[sentence_id][1],user_motor_aoi_id[sentence_id],user_motor_aoi_center_tuple[sentence_id][0],user_motor_aoi_center_tuple[sentence_id][1],user_workload[sentence_id],user_curiosity[sentence_id],user_valid_focus[sentence_id],user_course_follow[sentence_id],user_engagement[sentence_id],user_confusion[sentence_id]]
                try:
                    agent_dur_result_list = [agent_gaze_aoi_id[sentence_id],agent_gaze_aoi_center_tuple[sentence_id][0],agent_gaze_aoi_center_tuple[sentence_id][1],agent_motor_aoi_id[sentence_id],agent_motor_aoi_center_tuple[sentence_id][0],agent_motor_aoi_center_tuple[sentence_id][1],agent_workload[sentence_id],agent_curiosity[sentence_id],agent_valid_focus[sentence_id],agent_course_follow[sentence_id],agent_engagement[sentence_id],agent_confusion[sentence_id]]
                except:
                    print(sentence_id,agent_gaze_aoi_id,agent_gaze_aoi_center_tuple,agent_motor_aoi_id,agent_motor_aoi_center_tuple,agent_workload,agent_curiosity,agent_valid_focus,agent_course_follow,agent_engagement,agent_confusion)
                    assert 1==0
                metric_ind_dur_result_list = [gaze_aoi_accuracy[sentence_id],gaze_aoi_distance[sentence_id],motor_aoi_accuracy[sentence_id],motor_aoi_distance[sentence_id],workload_diff[sentence_id],curiosity_diff[sentence_id],valid_focus_diff[sentence_id],follow_ratio_diff[sentence_id],engagement_accuracy[sentence_id],confusion_accuracy[sentence_id]]

                with open(self.sim_result_dur_path,'a+') as fwrite_ind:
                    write_list = self.sim_config_set + [str(self.agent_id),str(transcript_id),str(sentence_id)] + user_dur_result_list + agent_dur_result_list + metric_ind_dur_result_list
                    fwrite_ind.write(self._list_to_str_line(write_list))
            
            for question_id in question_id_list:
                metric_ind_question_result_list = [choice_similarity_item[question_id],accuracy_similarity_item[question_id]]
                with open(self.sim_result_post_path,'a+') as fwrite_ind:
                    write_list = self.sim_config_set + [str(self.agent_id),str(question_id)] + [user_answer[question_id],agent_answer[question_id],correct_answer[question_id]] + metric_ind_question_result_list
                    fwrite_ind.write(self._list_to_str_line(write_list))

            self.exist_simulation_transcript_id_list.append(transcript_id)

    
    def agent_run_all(self):
        self._store_log('\n\n Start Simulation for Agent '+str(self.agent_id)+' in Video: '+str(self.video_id)+'-'*40+'\n\n')
        self.instantiate_profile()
        self.instantiate_memory()
        
        self.agent_run_during_class()
        # self.agent_run_post_class()

    def _store_log(self,input_string,color=None, attrs=None, print=False):
        with open(self.log_file, 'a') as f:
            f.write(input_string + '\n')
            f.flush()
        if(print):
            cprint(input_string, color=color, attrs=attrs)

    def _calculate_mse(self,diff_list):
        # diff_list_filter = self._remove_none_from_list(diff_list)
        return sum(x**2 for x in diff_list) / len(diff_list)

    def _calculate_difference(self,truth,prediction):
        if truth == None or prediction == None:
            return None
        return abs(truth-prediction)

    def _calculate_same(self,user_truth,agent_prediction):
        if user_truth == None or agent_prediction == None:
            return None
        if user_truth == agent_prediction:
            return 1
        else:
            return 0

    def _calculate_distance(self,point_1_tuple,point_2_tuple):
        if point_1_tuple == None or point_2_tuple == None or point_1_tuple[0] == None or point_1_tuple[1] == None or point_2_tuple[0] == None or point_2_tuple[1] == None:
            return None 
        return math.sqrt((point_1_tuple[0]-point_2_tuple[0])*(point_1_tuple[0]-point_2_tuple[0])+(point_1_tuple[1]-point_2_tuple[1])*(point_1_tuple[1]-point_2_tuple[1]))

    def _calculate_accuracy(self,true_labels,predicted_labels):
        if len(true_labels) != len(predicted_labels):
            raise ValueError("Input lists must have the same length.")
        compare_list = []
        for comi in range(true_labels):
            if true_labels[comi] != None and predicted_labels[comi] != None:
                compare_list.append(int(true_labels[comi]==predicted_labels[comi]))
            else:
                compare_list.append(None)
        # compare_list = [int(true == pred) for true, pred in zip(true_labels, predicted_labels)]
        correct_predictions = sum(compare_list)
        accuracy = correct_predictions / len(compare_list)
        return accuracy,compare_list

    def _remove_none_from_list(self,input_list):
        filtered_list = [value for value in input_list if value is not None]
        return filtered_list

    def _list_to_str_line(self,input_list):
        return ','.join(str(u_value) for u_value in input_list)+'\n'



def simulate_student(student_id, agent_config):
    print('Simulating student:', student_id)
    avatar_item = Avatar(agent_config, agent_id=student_id)
    avatar_item.agent_run_all()

async def run_exp(agent_config,user_idx_start,user_idx_end):
    demo_table_origin = pd.read_csv(agent_config['dataset_path']+'/student_demo.csv')
    demo_table_extend = pd.read_csv(agent_config['dataset_path']+'/student_demo_generated.csv')
    student_id_list_origin = list(set(demo_table_origin['student_id']))
    student_id_list_extend = list(set(demo_table_extend['student_id']))
    student_id_list = student_id_list_origin if agent_config['memory_source'] == 'real' else student_id_list_extend
    student_id_list.sort()
    student_num = len(student_id_list)
    example_user_list = list(set([element for sublist in agent_config['example_user_dict'].values() for element in sublist]))

    
    student_list_valid = [e_user for e_user in student_id_list if e_user not in example_user_list]
    student_list_select = student_id_list[user_idx_start:user_idx_end]
    
    print('student num: ',len(student_list_select))

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=100)

    tasks = []

    for student_id in student_list_select:
        if os.path.exists(agent_config['result_path']) == False:
            os.makedirs(agent_config['result_path'])
        
        root_folder = agent_config['result_path'] + '/' + agent_config['memory_source'] + '_' + agent_config['forget_effect'] + '_reflect-' + agent_config['reflection_choice'] + '_' + agent_config['memory_component_choice'] + '_' + agent_config['sim_strategy'] + '_example-' + agent_config['example_demo'] + '_' + str(agent_config['gpt_type'])
        
        if os.path.exists(root_folder) == False:
            os.makedirs(root_folder)
        if os.path.exists(root_folder + '/log') == False:
            os.makedirs(root_folder + '/log')
        if os.path.exists(root_folder + '/agent_memory') == False:
            os.makedirs(root_folder + '/agent_memory')
        if os.path.exists(root_folder + '/user_memory') == False:
            os.makedirs(root_folder + '/user_memory')
        task = loop.run_in_executor(executor, simulate_student, student_id, agent_config)
        tasks.append(task)

    await asyncio.gather(*tasks)



# Memory module: PM: gaze. MM: motor. CM: cognitive state. KM: knowledge/course material. MUST include KM otherwise the memory does not make sense.

simulation_config_option = {
    'memory_source_list': ['real','sim'], 
    'reflection_list': ['yes','no'],
    'forget_list': ['no_memory','random_half_plus_recent_one','all_plus_recent_one','only_recent_one'],
    'memory_component_list': ['KM','KM+PM','KM+PM+MM','KM+PM+MM+CM'],
}


simulation_config_1 = {
    'dataset_path': 'dataset/',
    'result_path': 'dataset/simulation/test',
    'memory_source': 'sim', # ['real','sim'] real for exp 1, sim for exp 2
    'sim_strategy': 'standard_cog', # ['standard', 'standard_cog']
    'example_demo': 'no', # remove example demonstration of other students. 
    'gpt_type': 0, # [0,1,2,3,4], 0:gemini, 1:llama2-7b, 2:llama2-70b, 3:gpt3.5, 4:gpt4
    'reflection_choice': 'no', # remove additional reflection module which is not useful
    'forget_effect': 'only_recent_one', # ['only_recent_one','no_memory'] Only use few-shot memory in the last recent slide for personalized example demonstration.
    'memory_component_choice': 'KM+PM+MM+CM', # ['KM+PM+MM+CM','KM+PM+MM','KM+PM+CM','KM+MM+CM']
    'example_user_dict': {'video_1': [167, 179, 153], 'video_2': [590, 321, 327], 'video_3': [366, 349, 342], 'video_4': [436, 729, 696], 'video_5': [798, 789, 507]}, # This code is not useful right now.
}

user_start_index = 0
user_end_index = 1
asyncio.run(run_exp(simulation_config_1,user_start_index,user_end_index))






