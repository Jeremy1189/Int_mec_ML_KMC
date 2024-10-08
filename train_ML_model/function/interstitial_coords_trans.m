function [interstitial_ID_nn_set,interstitial_ID_count_set,relative_interstitial_coords] ...
    = interstitial_coords_trans(interstitial_coord,interstitial_coords,interstitial_index,lattice_constant,supersize)
% Description:
%       This function aims to obtain the atomIDs in nn shells
% input:
%      vac_coord:size=[1,3]
%       per_coords: size=[n,3]

% output:
%       vac_ID_nn_set:
%       vac_ID_count_set:
% time:
%          2020/11/11 (first version)
%%
relative_interstitial_coords = interstitial_coords - repmat(interstitial_coord,[length(interstitial_coords),1]);
fix_point_trans=1e10;
relative_interstitial_coords= round(relative_interstitial_coords.*fix_point_trans)./fix_point_trans;
% periodic boundary check
min_per_coords= zeros(1,3);
% lattice_constant=3.522;
% supersize=10;
max_per_coords=lattice_constant*supersize*ones(1,3);
box_size =  max_per_coords-min_per_coords;
upper_boundary = round(1/2.*(box_size).*fix_point_trans)./fix_point_trans;
% upper_boundary = round(1/2.*(box_size).*fix_point_trans)./fix_point_trans;
% err_tol =10^-2;
for i= 1:length(box_size)
    vac_index_modify= find(relative_interstitial_coords(:,i)>=upper_boundary(i));

    relative_interstitial_coords(vac_index_modify,i)= relative_interstitial_coords(vac_index_modify,i)-box_size(i);

    vac_index_modify= find(relative_interstitial_coords(:,i)<-upper_boundary(i));
  
    relative_interstitial_coords(vac_index_modify,i)= relative_interstitial_coords(vac_index_modify,i)+box_size(i);
   
end
% calculated the distance
vac_nn_distance = sqrt( sum(relative_interstitial_coords.^2,2) );

% fixed
fixed_point=8;
vac_nn_distance = round(vac_nn_distance*10^fixed_point)/10^fixed_point;

%unique distance
%vac
unique_vac_dis_set= unique(vac_nn_distance);
L_unique = length(unique_vac_dis_set);
% largest_nn=L_unique;
largest_nn=2;
interstitial_ID_nn_set =zeros(largest_nn,1);% output1
interstitial_ID_count_set= zeros(largest_nn,1);% remove vacancy itself distance 
count_vac=0;

for num_unique = 1:largest_nn
    %vac 
%     index_vac_set=find(vac_nn_distance ==unique_vac_dis_set(num_unique+1));
    index_vac_set=find(vac_nn_distance ==unique_vac_dis_set(num_unique));
    interstitial_ID_count_set(num_unique) = length(index_vac_set);
    interstitial_ID_nn_set(count_vac+1 : count_vac+ length(index_vac_set) ) = interstitial_index(index_vac_set);%output2
    count_vac = count_vac + length(index_vac_set);  
end

end%end function

