% clc;
clear;
addpath(genpath(pwd));
rng('default')
%% load the ML model
load optimal_model.mat optimal_model
ML_model=optimal_model;% input[NN,num_samples]
%% initialization
load('load_Data/per55');
lattice_cos_ini= per55(5,3);
per_index=per55(:,1);
per_type=per55(:,2);
per_coords=per55(:,3:5);

fix_point=1e10;
frac_per=round((per_coords./lattice_cos_ini).*fix_point)./fix_point;
% per_index = perfect_structure(1,:);
% per_coords =  perfect_structure(3:5,:);

NN1=12;%fcc 1st nn
ratio_temp=(0:0.1:1)';
ratio=[ratio_temp,1-ratio_temp];
lattice_constant_set_Ni =[3.4989,3.4919, 3.4888,3.4872,...
    3.4874,3.4882,3.4923,3.4945,3.4996,...
    3.5077,3.5195];
super_size_length=10;
set_dis=1.2;
Kb= 0.025852/300 ;% ev
D0=1e15;%HZ
T=1000;
L_size=1e6;
parfor num_ratio=1:length(ratio)
    current_ratio=ratio(num_ratio,:);
    lattice_constant=lattice_constant_set_Ni(num_ratio);
%%   creating interstitial structure
    insert_pos_atom_ID=2221;    
    perfect_structure_coords=frac_per.*lattice_constant;
    perfect_structure=[per_index,per_type,perfect_structure_coords];
    % insert_pos_atom_ID=1;% boundary test
    insert_interstitial_type=perfect_structure(insert_pos_atom_ID,2);       
    
    L=lattice_constant*super_size_length;
    
    boundary = [0,L;0,L;0,L];
    
    interstitial_structure=creat_dumbell_interstitial(perfect_structure,...
        insert_pos_atom_ID, insert_interstitial_type,set_dis, boundary);
    
    %% the standard coordinates
    initerstitial_per_coords= interstitial_structure(:,3:5);
    interstitial_per_index=interstitial_structure(:,1);
    initia_I1_ID_back=interstitial_structure(end,1);
    initia_I2_ID_mig=insert_pos_atom_ID;% the insert atom set as the last index in interstitial_structure    
    final_mig_I3_ID=2222;% this ID must the 1st NN of the insert_pos_atom
    standard_transformed_coords =coord_transform(initerstitial_per_coords,initia_I1_ID_back,initia_I2_ID_mig,...
        final_mig_I3_ID,lattice_constant,super_size_length);
    %% the random type according to the ratio
    element_type_set=1:length(current_ratio);
    interstitial_per_types= random_ratio_type(length(initerstitial_per_coords),element_type_set,current_ratio);
    interstitial_per_types=interstitial_per_types';
%     interstitial_structure(:,2)=random_ratio_type_set;
    type_index_cell=cell(1,length(element_type_set));
    for num_type=1:length(element_type_set)
        curent_index= find(interstitial_per_types==element_type_set(num_type));
        type_index_cell{num_type}=curent_index;
    end
    
    %% caluculation the nearest neibours, obtain the sort ID
    % this ID sort must satisfy the same rule of the  coord_transform
    % function, that has a same origin 
    central_position_mig_former=0.5.*( initerstitial_per_coords(initia_I1_ID_back,:)+initerstitial_per_coords(initia_I2_ID_mig,:));% the dumbbell central before migration
    central_position_mig_former=boundary_check(central_position_mig_former,boundary);
    central_position_mig_latter =initerstitial_per_coords(final_mig_I3_ID,:); % the new I1 and I3 will generate a new central position
    NN_num1=5;
    NN_num2=NN_num1+2;
    
    [central1_ID_nn_set,central1_ID_count_set,relative_central1_coords]...
        = update_interstitial_nn_kmc(central_position_mig_former,initerstitial_per_coords,lattice_constant,super_size_length,NN_num1);
%     fraction1_coords=relative_central1_coords./lattice_constant;
    [central2_ID_nn_set,central2_ID_count_set,relative_central2_coords]...
        = update_interstitial_nn_kmc(central_position_mig_latter,initerstitial_per_coords,lattice_constant,super_size_length,NN_num2);
    sort_ID=unique([central1_ID_nn_set;central2_ID_nn_set]);
    %% input relative coordinates
    input_standard_coords=standard_transformed_coords(sort_ID,:);
    %% kMC initial
    
    R0=interstitial_structure(:,3:5);
    Rt=R0;
    cross_boundary_record=zeros(size(R0));
    
    tracks= zeros(L_size,12);%[t,I1_x,I1_y,I1_z,I2_x,I2_y,I2_z,MSD], I1 is the mig interstitial, I2 is the back interstitial
    mig_index_set = zeros(L_size,1);
    k_tot_avg=zeros(L_size,1);
    NN1_mig_ID_count_set=zeros(L_size,8);    
    NN1_mig_energy_count_set=zeros(L_size,8);
    t=0;
   
    %% kMC loop
    interstitial_ID_back=initia_I1_ID_back;
    interstitial_ID_mig=initia_I2_ID_mig;
    for count=1:L_size
        
        if mod(count,1e4)==0
            disp(count)
        end
        % calculating MSD
        Dt =(Rt-R0);
        
        Dt=Dt+L.* cross_boundary_record;
        sum_Dt=sum(Dt.^2,2);%
        %              sum_Dt(vacancy_atomID)=0;
        N=length(interstitial_per_types);
        MSD =sum(sum_Dt)./N;
        %partitial MSD
        MSD_element=zeros(1,length(type_index_cell));
        for num_types=1:length(type_index_cell)
            current_index=type_index_cell{num_type};
            if ~isempty(current_index)
                sum_cur=sum_Dt(current_index);
                %                      r_index=find(Co_index==vac_ID);
                MSD_element(num_type)=sum(sum_cur)./length(current_index);
            else
                MSD_element(num_type)=0;
            end
        end
        interstitial_back_coord = initerstitial_per_coords(interstitial_ID_back,:);
        interstitial_mig_coord = initerstitial_per_coords(interstitial_ID_mig,:);
        tracks(count,:)=[t,interstitial_ID_back,interstitial_back_coord,...
            interstitial_ID_mig,interstitial_mig_coord,MSD,MSD_element];
        %% obtaining the migtration energy of the 1st NN of the two interstitial
        largest_nn=1;
        % I1 has 4 possible migration paths
        [central1_ID_nn_set,central1_ID_count_set,~] ...
            = update_interstitial_nn_kmc(interstitial_back_coord,initerstitial_per_coords,lattice_constant,super_size_length,largest_nn+1);% the nearest atom is itself
        NN_I1_ID=central1_ID_nn_set(2:central1_ID_count_set(2)+1);% 2:central1_ID_count_set(2)+1 is the total 1st nn atoms
        interstitial_back_energy=zeros(1,length(NN_I1_ID));
        for num_mig =1:length(NN_I1_ID)
            initia_I1_ID_back= interstitial_ID_mig;
            initia_I2_ID_mig= interstitial_ID_back;
            final_mig_I3_ID = NN_I1_ID(num_mig);
            transformed_coords = coord_transform(initerstitial_per_coords,initia_I1_ID_back,initia_I2_ID_mig,...
                final_mig_I3_ID,lattice_constant,super_size_length);
            %sort ID
            dif_vector =initerstitial_per_coords(initia_I1_ID_back,:)-initerstitial_per_coords(initia_I2_ID_mig,:);
            dif_vector =dis_boundary_check(dif_vector,boundary);
            central_position_mig_former=initerstitial_per_coords(initia_I2_ID_mig,:)+0.5.*dif_vector;% the dumbbell central before migration
            central_position_mig_former=boundary_check(central_position_mig_former,boundary);
            central_position_mig_latter =initerstitial_per_coords(final_mig_I3_ID,:); % the new I1 and I3 will generate a new central position
            % the NN input of the migration before and after
            [central1_ID_nn_set,~,~]...
                = update_interstitial_nn_kmc(central_position_mig_former,initerstitial_per_coords,lattice_constant,super_size_length,NN_num1);
            
            [central2_ID_nn_set,~,~]...
                = update_interstitial_nn_kmc(central_position_mig_latter,initerstitial_per_coords,lattice_constant,super_size_length,NN_num2);
            current_ID_set=unique([central1_ID_nn_set;central2_ID_nn_set]);
            current_coords_set = transformed_coords(current_ID_set,:);
            % sort the ID
            [C,index_current,index_input] = intersect(current_coords_set,input_standard_coords,'rows');%C= current_coords_set(index_current,:) and C = input_standard_coords(index_input,:)
            [sort_value, sort_standart_input_index]= sort(index_input);
            % adjust the current_ID has a same order with the standard one
            current_sort_ID=current_ID_set(sort_standart_input_index);% current_coords corresponds to the current_ID_set
            % predict the energy
            current_input = interstitial_per_types(current_sort_ID);
            interstitial_back_energy(num_mig)= ML_model(current_input);
        end
        
        % I2 has another 4 possible migration paths
        
        [central2_ID_nn_set,central2_ID_count_set,relative_central2_coords] ...
            = update_interstitial_nn_kmc(interstitial_mig_coord,initerstitial_per_coords,lattice_constant,super_size_length,largest_nn+1);
        NN_I2_ID=central2_ID_nn_set(2:end);
        interstitial_mig_energy=zeros(1,length(NN_I2_ID));
        for num_mig =1:length(NN_I2_ID)
            initia_I1_ID_back= interstitial_ID_back;
            initia_I2_ID_mig= interstitial_ID_mig;
            final_mig_I3_ID = NN_I2_ID(num_mig);
            transformed_coords = coord_transform(initerstitial_per_coords,initia_I1_ID_back,initia_I2_ID_mig,...
                final_mig_I3_ID,lattice_constant,super_size_length);
            %sort ID
             dif_vector =initerstitial_per_coords(initia_I1_ID_back,:)-initerstitial_per_coords(initia_I2_ID_mig,:);
            dif_vector =dis_boundary_check(dif_vector,boundary);
            central_position_mig_former=initerstitial_per_coords(initia_I2_ID_mig,:)+0.5.*dif_vector;% the dumbbell central before migration
            central_position_mig_former=boundary_check(central_position_mig_former,boundary);           
            
            central_position_mig_latter =initerstitial_per_coords(final_mig_I3_ID,:); % the new I1 and I3 will generate a new central position
            % the NN input of the migration before and after
            [central1_ID_nn_set,~,~]...
                = update_interstitial_nn_kmc(central_position_mig_former,initerstitial_per_coords,lattice_constant,super_size_length,NN_num1);
            
            [central2_ID_nn_set,~,~]...
                = update_interstitial_nn_kmc(central_position_mig_latter,initerstitial_per_coords,lattice_constant,super_size_length,NN_num2);
            current_ID_set=unique([central1_ID_nn_set;central2_ID_nn_set]);
            current_coords_set = transformed_coords(current_ID_set,:);
            % sort the ID
            [C,index_current,index_input] = intersect(current_coords_set,input_standard_coords,'rows');%C= current_coords_set(index_current,:) and C = input_standard_coords(index_input,:)
            [sort_value, sort_standart_input_index]= sort(index_input);
            % adjust the current_ID has a same order with the standard one
            current_sort_ID=current_ID_set(sort_standart_input_index);% current_coords corresponds to the current_ID_set
            % predict the energy
            current_input = interstitial_per_types(current_sort_ID);
            interstitial_mig_energy(num_mig)= ML_model(current_input);
        end
        NN1_mig_ID_set=[NN_I1_ID;NN_I2_ID]';
        NN1_mig_ID_count_set(count,:)=NN1_mig_ID_set;
        NN1_mig_energy_set=[interstitial_mig_energy,interstitial_mig_energy];
        NN1_mig_energy_count_set(count,:)= NN1_mig_energy_set;
        %             %             NN1_energy_set(count,:)= NN1_mig_energy_set;
        %% kMC
        K_set= D0.*exp(-(NN1_mig_energy_set)./(Kb*T));
        K_tot =sum(K_set);
        k_tot_avg(count)=K_tot;
        cum_k_set= cumsum(K_set);
        roulette_k_set = cum_k_set./K_tot;
        r1 =rand(1);
        mig_index = find(r1-roulette_k_set <0,1);
        r2 =rand(1);
        t = t + -1/K_tot* log(r2);
        
        %% update the interstitial coordinate
        target_interstitial_ID = NN1_mig_ID_set(mig_index);
        mig_index_set(count)=mig_index;
        if mig_index>length(NN1_mig_ID_set)/2
            migration_interstitial_coord = initerstitial_per_coords(interstitial_ID_mig,:);
        else
            migration_interstitial_coord = initerstitial_per_coords(interstitial_ID_back,:);
        end
        %% there are six possiple position for forming a new dumbell
        %interstitiall pair
        new_interstitial_old_coord = initerstitial_per_coords(target_interstitial_ID,:);
        possible_pos=[new_interstitial_old_coord+[set_dis,0,0];...
            new_interstitial_old_coord+[-set_dis,0,0];...
            new_interstitial_old_coord+[0,set_dis,0];...
            new_interstitial_old_coord+[0,-set_dis,0];...
            new_interstitial_old_coord+[0,0,set_dis];...
            new_interstitial_old_coord+[0,0,-set_dis]];
        % boundary check
        for s=1:length( possible_pos)
            possible_pos(s,:)=boundary_check( possible_pos(s,:),boundary);
        end
        dif_dis_set= possible_pos-repmat(migration_interstitial_coord,size(possible_pos,1),1);%6 possible positions
        dif_dis_set= dis_boundary_check(dif_dis_set, boundary);
        sum_dif_dis= sum(dif_dis_set.^2,2);
%         sum_dis_set= sum(sum_dif_dis);
        [~,min_index]=min(sum_dif_dis);
        new_interstitial_new_coord=possible_pos(min_index,:);
        interstitial_new_I1_coord= boundary_check(new_interstitial_new_coord,boundary);% final position of the migration atom 
        % the new target position
        direction_distance=new_interstitial_old_coord-new_interstitial_new_coord;
        direction_distance= dis_boundary_check(direction_distance, boundary);
        interstitial_new_I2_coord= new_interstitial_new_coord + 2*direction_distance;
        interstitial_new_I2_coord= boundary_check(interstitial_new_I2_coord,boundary);% final position of the target atom (to form the new interstitial)
        % update the interstitial per_coords and IDs
        initerstitial_per_coords(target_interstitial_ID,:)=interstitial_new_I2_coord;
        if mig_index>length(NN1_mig_ID_set)/2
            temp_coord =initerstitial_per_coords(interstitial_ID_mig,:);
            initerstitial_per_coords(interstitial_ID_mig,:)=interstitial_new_I1_coord;% uptate the migration atom
            dif_int_coord = temp_coord-initerstitial_per_coords(interstitial_ID_back,:);
            dif_int_coord=dis_boundary_check(dif_int_coord,boundary);
            initerstitial_per_coords(interstitial_ID_back,:)=initerstitial_per_coords(interstitial_ID_back,:)+dif_int_coord/2;
            initerstitial_per_coords(interstitial_ID_back,:)= boundary_check( initerstitial_per_coords(interstitial_ID_back,:),boundary);
            interstitial_ID_back =target_interstitial_ID;% update the interstitial ID
            
        else
            temp_coord=initerstitial_per_coords(interstitial_ID_back,:);
            initerstitial_per_coords(interstitial_ID_back,:)=interstitial_new_I1_coord;% uptate the migration atom
             dif_int_coord =temp_coord-initerstitial_per_coords(interstitial_ID_mig,:);
            dif_int_coord=dis_boundary_check(dif_int_coord,boundary);
            initerstitial_per_coords(interstitial_ID_mig,:)=initerstitial_per_coords(interstitial_ID_mig,:)+dif_int_coord/2;
            initerstitial_per_coords(interstitial_ID_mig,:)= boundary_check( initerstitial_per_coords(interstitial_ID_mig,:),boundary);
            interstitial_ID_mig =target_interstitial_ID;% update the interstitial ID
        end
         
        
        %% cross boundary statistic
        initial_pos=Rt;
        final_pos=initerstitial_per_coords;
        cross_boundary_record = cross_boundary_statistic(initial_pos,final_pos,boundary,cross_boundary_record);
        Rt=initerstitial_per_coords;
    end
    
    
  
    %% plot
    L_tr=length(tracks);
        t_set  =tracks(1:L_tr,1);
        MSD_set  =tracks(1:L_tr,10)*1e-20;       
        x=t_set;
%         figure;
%         scatter(x,MSD_set,'.')
        P=polyfit(x,MSD_set,1);
%         a=0:max(x)/1e3:max(x);
%         f = polyval(P,a);
%         hold on;
%         plot(a,f,'r')
        D=P(1)./6;
%         disp(D(num_ratio))
%         xlabel('time(s)')
% %         ylabel('MSD ($\AA$)','interpreter','latex')
%          ylabel('MSD (m^2)')
%         title(['T=',num2str(T),'K'])
%         set(gca,'FontSize',12,'Fontname', 'Arial','FontWeight','bold',"LineWidth",2)
        %% store
          interstial_struct_cell{num_ratio}=[interstitial_per_index,interstitial_per_types,initerstitial_per_coords];
          R0_cell{num_ratio}= R0;
          Rt_cell{num_ratio}= Rt;
          cross_boundary_cell{num_ratio}= cross_boundary_record;
          tracks_cell{num_ratio}=tracks;
          k_tot_avg_cell{num_ratio}= k_tot_avg;
          mig_index_cell{num_ratio}=mig_index_set;
          NN1_mig_ID_count_cell{num_ratio}=NN1_mig_ID_count_set;
          NN1_mig_energy_cell{num_ratio}=NN1_mig_energy_count_set;
          D_cell{num_ratio}=D;
          
end
save (['dif_ratio_',num2str(T),'K.mat']);

%         %%
%         
%         
%     end
%     tracks_set{num_T}=tracks;
%     k_tot_avg_set{num_T}= k_tot_avg;
%     cos_set{num_T}=cos_value_set;
%     f_set(num_T)=(1+mean(cos_value_set))/(1-mean(cos_value_set));
%     D_set=[D_set;D.*(1e-20.*N)];
%     num_T =num_T+1;
%     %  save avg_origin_2w_t_900k_ratio_Ni_1_0.mat
%     %%
%     %     save(['avg_inters_Ni_Dt_',num2str(T),'K.mat'])
% end
% figure;
% semilogy(1000./(500:200:1300),D_set,'-o',"LineWidth",2)
% xlabel('1000/T(1/K)')
% ylabel('tracer diffusion coefficient(m^2/s)')
% title('Pure Ni')
% set(gca,'FontSize',12,'Fontname', 'Arial','FontWeight','bold',"LineWidth",2)
% %  D_avg=D;
%  save D_avg.mat D_avg ratio_Ni_order
%  figure;
% semilogy(1-ratio_Ni_order,D_avg.*(4000*1e-20),'-s',"LineWidth",2,'Color',	'#D95319')
% xlabel('Fe_xNi_1_-_x','FontSize',12,"FontWeight","bold")
% ylabel('Diffusion Rate(m^2/s)','FontSize',12,"FontWeight","bold")
% set(gca,'FontSize',12,'Fontname', FN,'FontWeight','bold');
% load  statistic_cos_t.mat
% figure ;
% plot(ratio_Fe,count_degree(:,1),'-p',ratio_Fe,count_degree(:,2),'-V',ratio_Fe,count_degree(:,3),'-s',...
%     ratio_Fe,count_degree(:,4),'-o',ratio_Fe,count_degree(:,5),'-^','LineWidth',2)
% legend({'cos0','cos60','cos90','cos120','cos180'})
% xlabel('Fe_xNi_1_-_x','FontSize',12,"FontWeight","bold")
% ylabel('Count','FontSize',12,"FontWeight","bold")
% set(gca,'FontSize',12,'Fontname', 'Arial','FontWeight','bold');
% title('Avg Potential')
% save statistic_cos_t.mat
% D_avg=D;
%  save D_avg500k.mat D_avg ratio_Fe
% save data1227.mat
% save k_tot_avg.mat k_tot_avg

% figure;
% for i= 1:length(k_tot_avg_set)
%     mean_t(i)= mean(k_tot_avg_set{i});
% end
% plot(ratio_Fe,1./mean_t,'-s',"LineWidth",2,'Color','r')
% xlabel('Fe_xNi_1_-_x','FontSize',12,"FontWeight","bold")
% ylabel('Average Time (s)','FontSize',12,"FontWeight","bold")
% set(gca,'FontSize',12,'Fontname', 'Arial','FontWeight','bold');
% title('Avg Potential')
% % %plot
%
% for s= 1:L_r
%     if num_ratio==1
%         figure;
%     else
%         hold on
%     end
%     t_set  =tracks_cell{s}(1:count_set(s),1);
%     MSD_set  =tracks_cell{s}(1:count_set(s),end);
%     us=10^6;
%     nm2=100;
%     plot(t_set'.* us,MSD_set./nm2,'-','LineWidth',1.5)
%     %  plot(t_set(1:1e4)'.*10^3,MSD_set(1:1e4)./100,'-','LineWidth',1.5)
%     %  plot(t_set(1:1e4)',MSD_set(1:1e4)./100,'-','LineWidth',1.5)
%     xlabel('Time(μs)')
%     %  xlabel('Time(ms)')
%     %  xlabel('Time(s)')
%     ylabel('MSD(nm^2)')
%     %         ylabel('MSD(A^2)')
%     title(['The MSD Vs Time for Ni_0_._',num2str(round(ratio(num_ratio)*10)),...
%         'Fe_0_._',num2str(round((1-ratio(num_ratio))*10)),'-900K'])
%     set(gca,'FontSize',14,'Fontname', 'Arial','FontWeight','bold');
%     x=t_set.*10^6;
%     P=polyfit(x,MSD_set/nm2,1);
%     a=0:10^-6:max(x);
%     f = polyval(P,a);
%     hold on;
%     plot(a,f,'r')
%     D(s)=P(1)./6;
% end


