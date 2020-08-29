%%%%%%%%%% Select dataset %%%%%%%%%%
flipped = true;
if flipped
	fname = 'multislice_test_crystal';
else
	fname = 'multislice_test_crystal_flipped';
end

arr = load(fname);
box = arr.box;
arr = arr.arr1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Useful parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% Load multem default parameter %%%%%%%%$$%%%%%%%%%
input_multislice = multem_default_values();          % Load default values;

%%%%%%%%%%%%%%%%%%%%%%% Specimen information %%%%%%%%%%%%%%%%%%%%%%%
input_multislice.spec_lx = box(1);
input_multislice.spec_ly = box(2);
input_multislice.spec_dz = 2;

%%%%%%%%%%%%%%%%%%%%%% x-y sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.nx = 500;
input_multislice.ny = 500;

%%%%%%%%%%%%%%%%%%%% Microscope parameters %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.E_0 = 300;                          % Acceleration Voltage (keV)
input_multislice.cond_lens_outer_aper_ang = 2.0;   % Outer aperture (mrad)
input_multislice.theta = 0.0;                       % Till ilumination (º)
input_multislice.phi = 0.0;                         % Till ilumination (º)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Standard parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%% Set system configuration %%%%%%%%%%%%%%%%%%%%%
system_conf.precision = 1;                           % eP_Float = 1, eP_double = 2
system_conf.device = 2;                              % eD_CPU = 1, eD_GPU = 2
system_conf.cpu_nthread = 1; 
system_conf.gpu_device = 0;

%%%%%%%%%%%%%%%%%%%% Set simulation experiment %%%%%%%%%%%%%%%%%%%%%
% eTEMST_STEM=11, eTEMST_ISTEM=12, eTEMST_CBED=21, eTEMST_CBEI=22, eTEMST_ED=31, eTEMST_HRTEM=32, eTEMST_PED=41, eTEMST_HCTEM=42, eTEMST_EWFS=51, eTEMST_EWRS=52, 
% eTEMST_EELS=61, eTEMST_EFTEM=62, eTEMST_ProbeFS=71, eTEMST_ProbeRS=72, eTEMST_PPFS=81, eTEMST_PPRS=82,eTEMST_TFFS=91, eTEMST_TFRS=92
input_multislice.simulation_type = 21;

%%%%%%%%%%%%%% Electron-Specimen interaction model %%%%%%%%%%%%%%%%%
input_multislice.interaction_model = 1;              % eESIM_Multislice = 1, eESIM_Phase_Object = 2, eESIM_Weak_Phase_Object = 3
input_multislice.potential_type = 2;                 % ePT_Doyle_0_4 = 1, ePT_Peng_0_4 = 2, ePT_Peng_0_12 = 3, ePT_Kirkland_0_12 = 4, ePT_Weickenmeier_0_12 = 5, ePT_Lobato_0_12 = 6

%%%%%%%%%%%%%%%%%%%%%%% Potential slicing %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.potential_slicing = 2;              % ePS_Planes = 1, ePS_dz_Proj = 2, ePS_dz_Sub = 3, ePS_Auto = 4

%%%%%%%%%%%%%%% Electron-Phonon interaction model %%%%%%%%%%%%%%%%%%
input_multislice.pn_model = 1;                       % ePM_Still_Atom = 1, ePM_Absorptive = 2, ePM_Frozen_Phonon = 3
input_multislice.pn_coh_contrib = 0;
input_multislice.pn_single_conf = 0;                 % 1: true, 0:false (extract single configuration)
input_multislice.pn_nconf = 10;                      % true: specific phonon configuration, false: number of frozen phonon configurations
input_multislice.pn_dim = 110;                       % phonon dimensions (xyz)
input_multislice.pn_seed = 300183;                   % Random seed(frozen phonon)


%%%%%%%%%%%%%%%%%%%%%% Specimen thickness %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.thick_type = 1;                     % eTT_Whole_Spec = 1, eTT_Through_Thick = 2, eTT_Through_Slices = 3
% input_multislice.thick = 10;   % Array of thickes (Å)

%%%%%%%%%%%%%%%%%%%%%% Illumination model %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.illumination_model = 1;             % 1: coherente mode, 4: Numerical integration
input_multislice.temporal_spatial_incoh = 1;         % 1: Temporal and Spatial, 2: Temporal, 3: Spatial

%%%%%%%%%%%%%%%%%%%%%%%%%%% Incident wave %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.iw_type = 2;                        % 1: Plane_Wave, 2: Convergent_wave, 3:User_Define, 4: auto
input_multislice.iw_psi = read_psi_0_multem(input_multislice.nx, input_multislice.ny);    % user define incident wave
input_multislice.iw_x = box(1)/2;  % x position 
input_multislice.iw_y = box(2)/2;  % y position

%%%%%%%%%%%%%%%%%%%%%%%% condenser lens %%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.cond_lens_m = 0;                   % Vortex momentum
input_multislice.cond_lens_c_10 = 1110;             % Defocus (Å)
input_multislice.cond_lens_c_30 = 0.0;              % Third order spherical aberration (mm)
input_multislice.cond_lens_c_50 = 0.00;             % Fifth order spherical aberration (mm)
input_multislice.cond_lens_c_12 = 0.0;              % Twofold astigmatism (Å)
input_multislice.cond_lens_phi_12 = 0.0;            % Azimuthal angle of the twofold astigmatism (º)
input_multislice.cond_lens_c_23 = 0.0;              % Threefold astigmatism (Å)
input_multislice.cond_lens_phi_23 = 0.0;            % Azimuthal angle of the threefold astigmatism (º)
input_multislice.cond_lens_inner_aper_ang = 0.0;    % Inner aperture (mrad) 

%%%%%%%%% defocus spread function %%%%%%%%%%%%
dsf_sigma = il_iehwgd_2_sigma(32); % from defocus spread to standard deviation
input_multislice.cond_lens_dsf_sigma = dsf_sigma;   % standard deviation (Å)
input_multislice.cond_lens_dsf_npoints = 5;         % # of integration points. It will be only used if illumination_model=4

%%%%%%%%%% source spread function %%%%%%%%%%%%
ssf_sigma = il_hwhm_2_sigma(0.45); % half width at half maximum to standard deviation
input_multislice.cond_lens_ssf_sigma = ssf_sigma;  	% standard deviation: For parallel ilumination(Å^-1); otherwise (Å)
input_multislice.cond_lens_ssf_npoints = 4;         % # of integration points. It will be only used if illumination_model=4

%%%%%%%%% zero defocus reference %%%%%%%%%%%%
input_multislice.cond_lens_zero_defocus_type = 1;   % eZDT_First = 1, eZDT_User_Define = 2
input_multislice.cond_lens_zero_defocus_plane = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = 1; n = 8; % precession angle alpha, 2^n images
% F is a single diffraction pattern imaged down the z-axis
% P(i,:,:) is the precessed image simulated with 2^(i-1) images
[F,P] = doPrecession(arr, n, alpha, system_conf, input_multislice);

% visualisation
m2psi_tot = squeeze(P(end,:,:));
m2psi_tot = log(1+1e4*m2psi_tot/max(m2psi_tot(:)));
I_min = min(m2psi_tot(:));I_max = max(m2psi_tot(:));
figure;imagesc(m2psi_tot, [I_min I_max]); 
axis image; colormap hot;
clear m2psi_tot I_min I_max i

