from pyswip import Prolog



def guess(character_dict):
    max_value = max(character_dict.values())
    character_list = [key for key, val in character_dict.items() if val == max_value]
    while len(character_list)!=1:
        print(f"Your character is {character_list[0]}? (Y/n)")
        answr = input()
        if answr.upper() =="N":
            character_list.pop(0)
        else:
            return character_list[0]
    return character_list[0]
    
def increment_values_by_keys(dictionary, keys, val=1):
    for key in keys:
        if key in dictionary:
            dictionary[key] += val
    return dictionary

def decrement_values_by_keys(dictionary, keys, val =1):
    for key in keys:
        if key in dictionary:
            dictionary[key] -= val
    return dictionary


def get_possible_characters_by_sex(sex):
    query = f"sex(Character, '{sex}')"
    res = list(prolog.query(query))
    return res

def get_possible_characters_by_role(role):
    query = f"role(Character, '{role}')"
    res = list(prolog.query(query))
    return res

def get_dangerous_characters():
    query = f"is_dangerous(X)"
    res = list(prolog.query(query))
    return res

def get_tricky_characters():
    query = f"can_trick(X)"
    res = list(prolog.query(query))
    return res

def get_main_antogonists_characters():
    query = f"main_antagonist(X)"
    res = list(prolog.query(query))
    return res

def get_all_characters():
    query = f"character(X)"
    res = list(prolog.query(query))
    chrc_dict = {val['X']:0 for  val in res}
    return chrc_dict

def ask_question(all_dict,question,get_method):
    characters = get_method()
    characters_list = [char['X'] for char in characters]
    print(question)
    answr = input()
    if answr.upper() =="Y":
        increment_values_by_keys(all_dict,characters_list)
    else:
        decrement_values_by_keys(all_dict,characters_list)

    # exit condition
    check_exit_condition(all_dict)


def is_unique_max_value(dictionary):
    if not dictionary:
        return False 

    max_value = max(dictionary.values())
    max_count = list(dictionary.values()).count(max_value)
    return max_count == 1

def check_exit_condition(characters_dict):
    if is_unique_max_value(characters_dict): 
        max_key = max(characters_dict, key=characters_dict.get)
        print(f"Your character is: {max_key}")
        exit(0)


def start_akinator():

    all_ch = get_all_characters()#all characters from knowledge base
    print("Welcome to the character Akinator!")

    # Get input from the user
    user_input = input("Enter the gender and role of the character (e.g., F Villain or M Hero): ").split()
    
    if len(user_input) != 2:
        print("Invalid input! Please enter the gender and role of the character (e.g., F Villain or M Hero).")
        return
    
    # check sex and role
    sex = user_input[0].upper()  # M or F
    role = user_input[1].capitalize()  # Villain, Hero, Ally, Antihero

    possible_characters_role = get_possible_characters_by_role(role)
    possible_characters_sex = get_possible_characters_by_sex(sex)
    characters_role_list = [char['Character'] for char in possible_characters_role]
    characters_sex_list = [char['Character'] for char in possible_characters_sex]
    increment_values_by_keys(all_ch,characters_role_list,val=10)
    increment_values_by_keys(all_ch,characters_sex_list, val=10)    

    # exit condition
    check_exit_condition(all_ch)
    
    ask_question(all_ch,"Your character is dangerous? (Y/n)",get_dangerous_characters)
    ask_question(all_ch,"Your character can trick? (Y/n)",get_tricky_characters)
    ask_question(all_ch,"Your character is main antogonist? (Y/n)",get_main_antogonists_characters)


    if not is_unique_max_value(all_ch):
        character_res = guess(all_ch)
        print(f"Your character is: {character_res}")
    else:
        print("Character not found.")


if __name__ == '__main__':
    prolog = Prolog()
    prolog.consult('task1.pl')
    start_akinator()