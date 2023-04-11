#%% deps

import openai
from re import sub
import os
from tqdm import tqdm
import threading

openai.api_key = os.environ['openai_api_key']


def get_chat_gpt_answer(chat_gpt_prompt):

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

#%% inference
if __name__ == '__main__':

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
    """
    all_objects = []

    for object in tqdm(sounding_objects):
         
        last_part_prompt = " variations of the sounding object " + str(object)

        current_variations = generate_objects(last_part_prompt,
                                        total_objects_wanted=variations_per_object_wanted,
                                        l_sounding_objects_per_batch=10,
                                        save_name=os.path.join('sounding_objects', object + "_variations.txt"),
                                        enable_tqdm=False)
        all_objects.extend(current_variations)
    """

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
