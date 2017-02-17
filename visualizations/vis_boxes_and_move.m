function [] = vis_boxes_and_move(scene_name)
%shows bounding boxes by image, with many options.  Can view vatic outputted boxes,
%results from a recognition system, or both. Also allows changing of vatic boxes. 
%
%INPUTS:
%         scene_name: char array of single scene name(not full path)


%TODO - enable more options to be set as input
%     - avoid reloading current image if invalid move

  init;%set path variables and MATLAB search path

  %%USER options
  show_depth = 0;
  show_instance_id = 0;



  %set paths for this scene
  scene_path =fullfile(ROHIT_BASE_PATH, scene_name);
  rgb_images_path = fullfile(scene_path,'jpg_rgb');
  depth_images_path = fullfile(scene_path,'high_res_depth');
  annotation_filename = fullfile(scene_path,'annotations.json');

  %load the annotations, and get all the image names 
  data = loadjson_wrapper(annotation_filename);
  image_names = keys(data);
 

  cur_image_name = image_names{1};
  next_image_name = '';
  move_command = ' '; 


  while(~strcmp(move_command,'q'))

    %load the current image, display it
    rgb_image = imread(fullfile(rgb_images_path,cur_image_name));
    hold off;
    imshow(rgb_image);
    hold on;

    %get the annotations for this image, and draw the boxes
    cur_annotation = data(cur_image_name);
    boxes = cur_annotation.bounding_boxes;
    for jl=1:size(boxes,1)
      bbox = double(boxes(jl,:));
      rectangle('Position',[bbox(1) bbox(2) (bbox(3)-bbox(1)) (bbox(4)-bbox(2))], ...
                   'LineWidth',3, 'EdgeColor','r');
      if(show_instance_id)
        t = text(bbox(1), bbox(2)-font_size,num2str(bbox(5)),  ...
                                    'FontSize',font_size, 'Color','white');
        t.BackgroundColor = 'red';
      end
    end%for jl, each  box
 
    %overlay the depth image if option is set 
    if(show_depth)
      depth_image = imread(fullfile(depth_images_path,...
                          strcat(cur_image_name(1:13), '03.png')));
     
      h = imagesc(depth_image);
      set(h,'AlphaData', .5);
    end%if show depth 
    
    %get the move command, and update the next_image_name accordingly
    move_command = input('Enter command: ', 's'); 

    
    if(move_command == 'q')
        disp('quiting...');
        break;
    elseif(move_command =='w')
        next_image_name = cur_annotation.forward;
    elseif(move_command =='a')
        next_image_name = cur_annotation.rotate_ccw;
    elseif(move_command =='s')
        next_image_name = cur_annotation.backward;
    elseif(move_command =='d')
        next_image_name = cur_annotation.rotate_cw;
    elseif(move_command =='e')
        next_image_name = cur_annotation.left;
    elseif(move_command =='r')
        next_image_name = cur_annotation.right;
    end


    %if the user inputted move is valid (there is an image there) 
    %then update the image to display. If the move was not valid, 
    %the current image will be displayed again
    if(~isempty(next_image_name))
      cur_image_name = next_image_name;
    end      


  end%while not quit
end%function
