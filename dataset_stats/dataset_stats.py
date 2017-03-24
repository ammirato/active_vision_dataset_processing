import os


def get_instance_name_to_id_dict(root):
    """
    Returns a dict with instance names as keys and is as values
    """
    name_to_id_dict = {} 
    for line in open(os.path.join(root,'instance_id_map.txt'),'r'):
        name, id_num = str.split(line)
        name_to_id_dict[name] = int(id_num)

    return name_to_id_dict


def get_instance_locations(root, instance_ids, scene_list=None):
    """
    Return which scenes each instance is present in

    ARGS:
        root (string): root directory of all scene directories
        istance_ids(List[int, ...]: the instances to consider

    KEYWORD ARGS:
        scene_list (List[string, ...]) = None:  Which scenes to consider.
                                                If not set, will consider
                                                all directories under root.

    RETURNS:
        dict: Keys: instance id
              Values: List of string scene names
    """
    #get the names of all the scenes
    if scene_list is None:
        scene_list = os.listdir(root)
        scene_list.sort()
    

    #to convert from string to id
    name_to_id_dict = get_instance_name_to_id_dict(root)

    #create dict to hold list of scenes each instance is present in
    instance_locations = {} 
    for id_num in instance_ids:
       instance_locations[id_num] = [] 

    scene_contains ={}
    for scene in scene_list:
        scene_contains[scene] = []
        present_names = []
        #load the present instance names
        for name in open(os.path.join(root,
                                      scene,
                                      'present_instance_names.txt'),
                         'r'):
       
             
            #lose the newline char and get the id 
            id_num = name_to_id_dict[name[:-1]]
        
            if id_num in instance_ids:
                instance_locations[id_num].append(scene)
                scene_contains[scene].append(id_num)
    
    return instance_locations , scene_contains




def get_set_with_most_elements(sets_dict,elements):
    """
    Returns set key that contains the most elements in the given list

    ARGS:
        sets_dict(Dict{}): keys: set names
                           values: List[] items  must be same type as items
                                   in elements List

        elements(List[]): items must have same type as sets_dict values items

    RETURNS:
        key of set name that has most elements 
    """
   
    max_set_name = None
    max_set_num = 0 
    for set_name in sets_dict.keys():
        num = len(set(elements).intersection(sets_dict[set_name]))
        if num > max_set_num:
            max_set_num = num
            max_set_name = set_name


    return max_set_name 
        




def get_two_set_covers(root, instance_ids, scene_list=None):
    """
    Attempts to return two small sets of scenes that cover all instances

    Not gaurenteed to find such sets if they exist.


    ARGS:
        root (string): root directory of all scene directories
        istance_ids(List[int, ...]: the instances to consider

    KEYWORD ARGS:
        scene_list (List[string, ...]) = None:  Which scenes to consider.
                                                If not set, will consider
                                                all directories under root.

    RETURNS:
        list with two list, one for each set cover: 
                                List[List[string, ...], List[string,...]]
    """


    #get the names of all the scenes
    if scene_list is None:
        scene_list = os.listdir(root)
        scene_list.sort()


    #
    _,scene_contains = get_instance_locations(root, instance_ids, scene_list=scene_list)

    not_covered = list(instance_ids)
    set_covers = [[],[]]

    for il in range(2):

        counter = 0
        while len(not_covered) > 0:

            max_set = get_set_with_most_elements(scene_contains,instance_ids) 
            set_covers[il].append(max_set)

            newly_covered = scene_contains[max_set]
            
            for id_num in newly_covered:
                if id_num in not_covered:
                    not_covered.remove(id_num)

            #clear out scene so it is not used twice
            scene_contains[max_set] = []
            counter += 1
            if counter == 100:
                print('No cover found')
                break

        #clear out scenes in the first cover so they are not used twice
        #also remove second scan of same scene
        for scene in set_covers[il]:
            scene_contains[scene] = []

            if scene[-1] == '1':
                other_scan = scene[:-1] + '2'
            else: 
                other_scan = scene[:-1] + '1'
            try:
                scene_contains[other_scan] = []
            except:
                nothing = 0


        #reset not covered
        not_covered = list(instance_ids)

    print set_covers









