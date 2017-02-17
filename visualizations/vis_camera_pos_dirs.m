function [] = vis_camera_pos_dirs(scene_name)
%show a figure with the camera positions plotted for a scene, 
%possibly also show a line coming from each point indicating the 
%orientation of the camera at that point
%
%INPUTS:
%         scene_name: char array of scene name, 


%TODO:   - make options optional input parameters 


  %initialize contants, paths and file names, etc. 
  init;

  %% USER OPTIONS
  view_direction = 1;%should the lines indicating camera direction be drawn?
  plot_type = 1; %  0 - 3D point plot 
                 %  1 - 2D point plot
  show_cluster_ids = 0; %plots number next to each point indicating which
                        %cluster it belongs to. A cluster is a group of 
                        %points that can be navigated using only roatation
  show_image_names = 0;%draw image name next to each point
  use_scaled_positions = 0;%use positions in meters, not arbitrary reconstruction coords


  %create figure
  positions_plot_fig = figure();

  %load camera position data 
  scene_path =fullfile(ROHIT_BASE_PATH, scene_name);
  image_structs_file =  load(fullfile(scene_path,'image_structs.mat'));
  image_structs = image_structs_file.image_structs;
  scale = image_structs_file.scale;


  %get positions and directions of camera for each image
  %scale the positions to be in millimeters is option is set
  %NOTE: USE OF scaled_world_pos FILED IS NOT RECOMMENED
  if(use_scaled_positions)
    world_poses = [image_structs.world_pos]*scale; 
  else
    world_poses = [image_structs.world_pos];
  end
  directions = [image_structs.direction];


  % plot positions 
  switch plot_type
    case 0 %use 3D plot
      plot3(world_poses(1,:),world_poses(2,:), world_poses(3,:),'r.');
    case 1 % make plot just 2D
      plot(world_poses(1,:),world_poses(3,:),'r.');
  end 

  %makes plot prettier
  axis equal;

  if(show_cluster_ids | show_image_names)
   
    cluster_ids = [image_structs.cluster_id]; 
    image_names = {image_structs.image_name};

    for jl=1:length(cluster_ids)
      cur_id = num2str(cluster_ids(jl)); 
      switch plot_type
        case 0 %use 3D plot
          if(show_cluster_ids)
            text(world_poses(1,jl),world_poses(2,jl), world_poses(3,jl),cur_id);
          end
        case 1 % make plot just 2D
          if(show_cluster_ids)
            text(world_poses(1,jl),world_poses(3,jl),cur_id,'FontSize', 10);
          end
          if(show_image_names)
            text(world_poses(1,jl),world_poses(3,jl),image_names{jl},'FontSize', 5);
          end
      end
    end%for jl, each images cluster id 
  end %if show cluster ids

  %plot direction arrows if option is set
  if(view_direction)
      hold on;

      switch plot_type
        case 0 %3D plot
          quiver3(world_poses(1,:),world_poses(2,:),world_poses(3,:), ...
             directions(1,:),directions(2,:),directions(3,:), ...
             'ShowArrowHead','off','Color' ,'b');
        case 1  %2D plot
          quiver(world_poses(1,:),world_poses(3,:), ...
             directions(1,:),directions(3,:), ...
             'ShowArrowHead','on','Color' ,'b');
      end%switch
      hold off;
  end%if view_direction

  %remove axis and make background white
  set(gca ,'visible', 'off');
  %set(gcf,'Color',[0,0,0]);
  set(gcf,'Color',[1,1,1]);

end%function



