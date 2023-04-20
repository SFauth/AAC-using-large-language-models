#%% deps

import openai
from re import sub
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import time



def get_chat_gpt_answer(chat_gpt_prompt, system_behavior_message=None):

    if system_behavior_message!=None:
           chat_gpt_response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[
                                                    {"role": "system", "content": system_behavior_message},
                                                    {"role": "user", "content": chat_gpt_prompt}
                                                ])       
           
    chat_gpt_response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[
                                                    {"role": "user", "content": chat_gpt_prompt}
                                                ])
                            
    return chat_gpt_response["choices"][0]["message"]["content"]


def clean_generated_sound_list(raw_answer_string):

    split_string = raw_answer_string.split("\n")

    # keep everything following a space, a full-stop / opening round bracket and a number

    pattern_1 = r"^\d+\. "
    pattern_2 = r"^\d+\) "

    after_pattern_1 = [sub(pattern_1, "", sound) for sound in split_string] 
    after_pattern_2 = [sub(pattern_2, "", sound) for sound in after_pattern_1]

    return after_pattern_2

def extend_AC_list_one_sample(prompt_first_part,
                   prompt_2nd_part,
                   keyword,
                   m_variations_wanted):
     

    sounding_objects = [] 
    chat_gpt_prompt = prompt_first_part + str(m_variations_wanted) + prompt_2nd_part + str(keyword)
    raw_answer_string = get_chat_gpt_answer(chat_gpt_prompt)
    sounding_objects.extend(clean_generated_sound_list(raw_answer_string))
    return sounding_objects


def extend_AC_list(keyword_list,
                   prompt_first_part,
                   prompt_2nd_part,
                   m_variations_wanted,
                   save_name=None):

    time.sleep(60)
    object_list_string = ", ".join(keyword_list)
    sounding_objects = [] 

    chat_gpt_prompt = prompt_first_part + str(m_variations_wanted) + prompt_2nd_part + object_list_string
    raw_answer_string = get_chat_gpt_answer(chat_gpt_prompt)
    sounding_objects.extend(clean_generated_sound_list(raw_answer_string))
    print("Current batch's objects generated")

    sounding_objects_no_duplicates = list(set(sounding_objects))

    if save_name != None:

        with open(save_name, 'w') as file:
                file.writelines("%s\n" % item for item in sounding_objects_no_duplicates)
    
    #queue_obj.put(sounding_objects_no_duplicates)

    return sounding_objects_no_duplicates


def generate_objects(last_part_prompt,
                      total_objects_wanted,
                        l_sounding_objects_per_batch,
                        save_name,
                        enable_tqdm=True):

    n_requests = total_objects_wanted / l_sounding_objects_per_batch

    chat_gpt_prompt = "Generate a list of " + str(l_sounding_objects_per_batch) + last_part_prompt

    if enable_tqdm==True:
        print ('ChatGPT prompt: {}'.format(chat_gpt_prompt))

    sounding_objects = []

    if enable_tqdm == True:

        for request in tqdm(range(int(n_requests))):

            raw_answer_string = get_chat_gpt_answer(chat_gpt_prompt)
            sounding_objects.extend(clean_generated_sound_list(raw_answer_string))

    else:

        for request in range(int(n_requests)):

            raw_answer_string = get_chat_gpt_answer(chat_gpt_prompt)
            sounding_objects.extend(clean_generated_sound_list(raw_answer_string))


    sounding_objects_no_duplicates = list(set(sounding_objects))

    if save_name != None:

        with open(save_name, 'w') as file:
                file.writelines("%s\n" % item for item in sounding_objects_no_duplicates)

    return sounding_objects_no_duplicates


# extend keywords with chat gpt



#%% inference
if __name__ == '__main__':

    print('Extend keyword list using ChatGPT...')      

    keywords = list(pd.read_csv("data/AudioSet/class_labels_indices.csv")["display_name"])
    keywords = [tag.strip() for tag in keywords for tag in tag.split(',')]

    print('Successfully read in initial keyword list')

    m_variations_wanted=3
    l_sounding_objects_per_batch=20
    
    def batch(keyword_list, batch_size):
        return [keyword_list[i:i+batch_size] for i in range(0, len(keyword_list), batch_size)]
    
    keywords_batched = batch(keywords,
                                l_sounding_objects_per_batch)        


    generated_keywords = Parallel(n_jobs=3)(
        delayed(extend_AC_list)(batch,
                                "Create ",
                                " variations\
                                of each object in the following list of audio tags, while\
                                each variation is not longer than 4 words! Do not skip!\
                                    list of audio tags=",
                                    m_variations_wanted) for batch in keywords_batched)
    
    save_name = "./data/sounding_objects/chatgpt_audio_tags_prompt_2.txt"

    with open(save_name, 'w') as file:
            file.writelines("%s\n" % item for item in generated_keywords)    

    import sys
    sys.exit()
    
    #%% clean data

    generated_keywords_list = [item for sublist in generated_keywords for item in sublist]
    separated_keywords = [keyword.split(",") for keyword in generated_keywords_list if "," in keyword]
    separated_keywords = [item for sublist in separated_keywords for item in sublist]

    listing_keywords_list = [keyword[2:] for keyword in generated_keywords_list if keyword.startswith("- ")]
    prompts_removed_keywords_list = [keyword.split(": ")[1] for keyword in separated_keywords if ": " in keyword]

    remaining_keywords = [keyword for keyword in generated_keywords_list if ":" not in keyword and not keyword.startswith("- ") and "," not in keyword]
    remaining_keywords.extend(prompts_removed_keywords_list)
    remaining_keywords.extend(listing_keywords_list)

    keywords.extend(remaining_keywords)

    ext_keywords = list(set(keywords))

    save_name = "./data/sounding_objects/extended_audioset_tags.txt"

    with open(save_name, 'w') as file:
            file.writelines("%s\n" % item for item in ext_keywords)    

    print('Printed file to txt!')


    """
    # Create initial sounding object list

    print ('Generate initial sounding obejct list ...')

    last_part_prompt =  " differing sounding objects that are not musical instruments that does not contain adjectives or verbs!"

    sounding_objects = generate_objects(last_part_prompt,
                                        total_objects_wanted=10,
                                        l_sounding_objects_per_batch=10,
                                        save_name=None
                                        )
    
    last_part_prompt =  " differing sounding objects that are musical instruments that does not contain adjectives or verbs!"

    musical_sounding_objects = generate_objects(last_part_prompt,
                                        total_objects_wanted=10,
                                        l_sounding_objects_per_batch=10,
                                        save_name=None
                                        )
        
    sounding_objects.extend(musical_sounding_objects)
    
    with open('all_initial_sounding_objects.txt', 'w') as file:
            file.writelines("%s\n" % item for item in sounding_objects)

    # Create variations of every object

    variations_per_object_wanted = 10
    
    all_objects = []

    for object in tqdm(sounding_objects):
         
        last_part_prompt = " variations of the sounding object " + str(object)

        current_variations = generate_objects(last_part_prompt,
                                        total_objects_wanted=variations_per_object_wanted,
                                        l_sounding_objects_per_batch=10,
                                        save_name=os.path.join('sounding_objects', object + "_variations.txt"),
                                        enable_tqdm=False)
        all_objects.extend(current_variations)
    

    threads = []

    for object in tqdm(sounding_objects):

        last_part_prompt = " variations of the sounding object " + str(object)

        t = threading.Thread(target=generate_objects, args=(last_part_prompt,
                                        variations_per_object_wanted,
                                        10,
                                        os.path.join('sounding_objects', object + "_variations.txt"),
                                        False))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    """
# %%
