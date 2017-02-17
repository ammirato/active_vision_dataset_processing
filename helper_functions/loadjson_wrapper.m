function [data] = loadjson_wrapper(filename)
%Loads annotations from the given filename(full path).
%Assumes format as in:
%http://cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.htm
%
%Uses json load function from matlab_file_exchange_helpers/json-lab...
%and adjusts the field names to match the image names
%
%
%INPUT 
%       filename - the full path to the json file to load
%
%OUtPUT 
%       data - a map from rgb image_name(with .jpg extension) to
%              a struct with the following fields:
%                 bounding_boxes: [Nx6]
%                 rotate_ccw: image_name
%                 rotate_ccw: image_name
%                 forward: image_name
%                 backward: image_name
%                 left: image_name
%                 right: image_name
%


  init;%set path variables and MATLAB search path

  %load the data using the file exchange library
  data_org = loadjson(filename);
  data = containers.Map();
  
  
  %get all the fieldnames. For some reason,
  %the jsonload function does something weird but
  %consistent with the image file names. 
  fieldnames = fields(data_org);

  %for each fieldname, fix it
  for il=1:length(fieldnames) 
    cur_name = fieldnames{il};

    split_name = strsplit(cur_name,'_');
    first_el= split_name{1};
    scene_type = first_el(end);
 
    image_name = strcat(scene_type,split_name{2},'.jpg');  
 
    data(image_name) = data_org.(cur_name);

  end%for il, each fieldname

end%function




