% Weathering Profile Simulation towards Steady State
% Peking University AI Assistant - Educational Version

clear; clc; close all;

%% Parameters Definition
% Physical constants
R = 8.314e-3; % Gas constant [kJ/mol/K]
T = 15 + 273.15; % Temperature [K]
T0 = 15 + 273.15; % Reference temperature [K]

% Profile parameters
n = 100; % Number of layers
h = 1; % Initial profile thickness [m]
dh = h / n; % Layer thickness [m]
S0 = 1.0; % Cross-sectional area [m?]

% Material properties
Ms = zeros(3, n); % Solid masses: [rock, weathered product, organic matter] [g]
poro = [0.05, 0.3, 0.1]; % Porosity for each material
den = [2.8e6, 2.2e6, 1.1e6]; % Density [g/m ]

% Preset weathering intensity
WI0 = 0.2;

% Calculate chemical weathering volume change rate
chem_vol_change_rate = (1/den(2)*0.6715 - 1/den(1)) * den(1);

% Particle properties
d = 0.001; % Particle diameter [m]

% Reaction parameters
m_ratio = 0.6715; % Mass ratio weathered/rock
n_CO2_rock = 0.00073; % CO2 moles per gram rock [mol/g]
n_CO2_org = 0.58/12; % CO2 moles per gram organic matter [mol/g]

% Rate constants
rw = 3e-3; % Weathering rate constant [g/m?/year]
a = 0.5; % CO2 sensitivity exponent
Ea = 60; % Activation energy [kJ/mol]
rox = 0.01; % Organic matter oxidation rate [1/year]

% Transport parameters
u = 1e-4; % Uplift rate [m/year], upward positive
s = 0; % liquid transport rate [m/year], downward positive
D0 = 0.1; % Base diffusion coefficient [m?/year]

% Boundary conditions
Cd_atm = 0.0114; % Atmospheric CO2 concentration [mol/m?]
hPOC = 0.2; % Organic matter input depth [m]
TPOCin = 0.1; % Total organic matter input [g/m?/year]
%TPOCin = 0; % Total organic matter input [g/m?/year]

% Numerical parameters
dt = 1.0; % Time step [year]
ddt = 1e-5; %Time step for dissolved [year]
tolerance = 1e-6; % Convergence tolerance (relative change)
max_iter = 1000000; % Maximum iterations

%% Initialization
% Initialize solid masses (only rock initially)
Ms(1,:) = den(1) * dh * S0; % Rock mass in each layer
Ms(2,:) = 0; % Weathered product
Ms(3,:) = 0; % Organic matter

% Initialize dissolved CO2 concentration
Cd = zeros(n, 1); % CO2 concentration [mol/m?]

% Calculate initial porosity for each layer
layer_poro = zeros(n, 1);
for j = 1:n
    total_volume = sum(Ms(:,j) ./ den');
    if total_volume > 0
        layer_poro(j) = sum((Ms(:,j) ./ den') .* poro') / total_volume;
    else
        layer_poro(j) = 0;
    end
end

% Storage for monitoring
time_record = [];
h_record = [];
profile_record = [];
volume_changes = [];
convergence_metrics = [];

%% Main Simulation Loop
t = 0;
converged = false;
iteration = 0;

fprintf('Starting simulation...\n');
fprintf('Time (yr)\tThickness (m)\tMax Rel Change\n');

while ~converged && iteration < max_iter
    iteration = iteration + 1;
    t = t + dt;
    
    % Store previous values for convergence check
    Ms_prev = Ms;
    Cd_prev = Cd;
    h_prev = h;
    
    %% Chemical Reactions
    
    for ii=1:max_iter
        % Calculate surface areas
        Sa = zeros(3, n);
        for j = 1:n
            for k = 1:3
                if den(k) > 0 && Ms(k,j) > 0
                    Sa(k,j) = Ms(k,j) / den(k) * (1 - poro(k)) / (d/3);
                end
            end
        end

        % Calculate reaction rates
        Reac_solid = zeros(3, n); % Solid reaction rates [g/year]
        Reac_dissolved = zeros(n, 1); % Dissolved reaction rates [mol/year]

        rate_factor = exp(-Ea/R * (1/T - 1/T0));

        for j = 1:n
            % Weathering reaction: rock + CO2 ¡ú weathered product
            if Ms(1,j) > 0 && Cd(j) > 0
                Rw = rw * Sa(1,j) * (Cd(j)/Cd_atm)^a * rate_factor;

                Reac_solid(1,j) = -Rw; % Rock consumption
                Reac_solid(2,j) = Rw * m_ratio; % Weathered product production
                Reac_dissolved(j) = Reac_dissolved(j) - Rw * n_CO2_rock/(dh*layer_poro(j)); % CO2 consumption
            elseif Cd(j)<0
                Rw = 0;
                Reac_solid(1,j) = -Rw; % Rock consumption
                Reac_solid(2,j) = Rw * m_ratio; % Weathered product production
                Reac_dissolved(j) = Reac_dissolved(j) - Rw * n_CO2_rock/(dh*layer_poro(j)); % CO2 consumption
            end

            % Organic matter oxidation: organic matter ¡ú CO2
            if Ms(3,j) > 0
                Rox = rox * Ms(3,j) * rate_factor;

                Reac_solid(3,j) = Reac_solid(3,j) - Rox; % Organic matter consumption
                Reac_dissolved(j) = Reac_dissolved(j) + Rox * n_CO2_org/(dh*layer_poro(j)); % CO2 production
            end
        end



        %% Dissolved Species Transport (Implicit Scheme)
        % Update diffusion coefficients
        D = D0 * layer_poro;
      %Set up differentiation operator matrix for CO2
        aa=-2*D/dh^2.*ones(n,1); %aa stores diagonal components
      for i=1:n-1
        bb(i+1)=D(i)/dh^2-s/2/dh; %bb stores array above, indexed by column
        cc(i+1)=D(i)/dh^2+s/2/dh; %cc stores array below, indexed by row
        aa(i)=aa(i);
      end
        aa(1)=-D(1)/dh^2-s/2/dh;%flux free at bottom
      %Set up forcing for CO2
        RHS=-Reac_dissolved;
        RHS(n)=RHS(n)-Cd_atm*(D(1)/dh^2+s/2/dh);
      %CO2 profile based on Ms guess

      %Use Thomas method to solve for tridiagonal system
        Cd_chem=thomas(aa,bb,cc,RHS);
        if min(Cd_chem)<0
            fprintf('Warning: dissolved negative value \n');
        end
        max_rel_change_Cdchem=max(abs(Cd_chem - Cd) ./ (abs(Cd) + eps));
        Cd=Cd_chem;
        if max_rel_change_Cdchem < tolerance
            break
        end
    end
    for j = 1:n
        % Weathering reaction: rock + CO2 ¡ú weathered product
        if Ms(1,j) > 0 && Cd(j) > 0
            Rw = rw * Sa(1,j) * (Cd(j)/Cd_atm)^a * rate_factor;

            Reac_solid(1,j) = -Rw; % Rock consumption
            Reac_solid(2,j) = Rw * m_ratio; % Weathered product production
            Reac_dissolved(j) = Reac_dissolved(j) - Rw * n_CO2_rock/(dh*layer_poro(j)); % CO2 consumption
        elseif Cd(j)<0
            Rw = 0;
            Reac_solid(1,j) = -Rw; % Rock consumption
            Reac_solid(2,j) = Rw * m_ratio; % Weathered product production
            Reac_dissolved(j) = Reac_dissolved(j) - Rw * n_CO2_rock/(dh*layer_poro(j)); % CO2 consumption
        end

        % Organic matter oxidation: organic matter ¡ú CO2
        if Ms(3,j) > 0
            Rox = rox * Ms(3,j) * rate_factor;

            Reac_solid(3,j) = Reac_solid(3,j) - Rox; % Organic matter consumption
            Reac_dissolved(j) = Reac_dissolved(j) + Rox * n_CO2_org/(dh*layer_poro(j)); % CO2 production
        end
    end  
  % Update solid masses due to chemical reactions
    Ms_chem = Ms + Reac_solid * dt;
    if min(Ms_chem)<0
        fprintf('Warning: Solid negative value \n');
    end
    Ms_chem = max(0, Ms_chem); % Ensure non-negative masses
%     for ii=1:max_iter*1000
%         % Set up implicit matrix for CO2 transport
%         A = zeros(n, n);
%         b = zeros(n, 1);
% 
%         for j = 1:n
%             % Diagonal element
%             A(j,j) = 1 + u*ddt/dh + 2*D(j)*ddt/(dh^2) + ...
%                 rw * Sa(1,j) * (Cd(j)/Cd_atm)^a * rate_factor* n_CO2_rock*...
%                 Cd(j)* ddt / (S0 * dh * (layer_poro(j) + eps));
% 
%             % Upwind element (advection)
%             if j < n
%                 A(j,j+1) = -u*ddt/dh;
%             end
% 
%             % Diffusion elements
%             if j > 1
%                 A(j,j-1) = A(j,j-1) - D(j)*ddt/(dh^2);
%             end
%             if j < n
%                 A(j,j+1) = A(j,j+1) - D(j)*ddt/(dh^2);
%             end
% 
%             % Source term: reactions
%             b(j) = Cd(j) + (Reac_dissolved(j)+rw * Sa(1,j) * ...
%                 (Cd(j)/Cd_atm)^a * rate_factor* n_CO2_rock) * ddt / (S0 * dh * (layer_poro(j) + eps));
%         end
% 
%         % Apply boundary conditions
%         % Top boundary (j = n): fixed concentration (atmospheric)
%         A(n,n) = A(n,n);
%         b(n) = b(n)+2*Cd_atm*D(n)*ddt/dh;
% 
%         % Bottom boundary (j = 1): no flux
%         A(1,1) = A(1,1) - D(1)*ddt/(dh^2);
% 
% 
%         % Solve linear system
%         Cd_new = A \ b;
%         Cd_r_change = max(abs(Cd - Cd_new) ./ (abs(Cd) + eps));
%         if Cd_r_change < tolerance
%     %     Cd_new = max(0, Cd_new); % Ensure non-negative concentrations
%             Cd=Cd_new;
%             break
%         end
%         Cd=Cd_new;
%     end
    %% Add Organic Matter Input
    Ms_temp = Ms_chem;
    
    if hPOC > h
        % Input to all layers (from top to bottom)
        POC_per_layer = TPOCin * dt / n;
        for j = 1:n
            Ms_temp(3,j) = Ms_temp(3,j) + POC_per_layer;
        end
    else
        % Input only to top layers within hPOC depth
        num_full_layers = floor(hPOC / dh);
        fractional_layer = hPOC / dh - num_full_layers;
        
        % Input to full layers (from top downward)
        if num_full_layers > 0
            POC_per_full_layer = TPOCin * dh / hPOC * dt;
            start_layer = n;
            end_layer = max(1, n - num_full_layers + 1);
            for j = start_layer:-1:end_layer
                Ms_temp(3,j) = Ms_temp(3,j) + POC_per_full_layer;
            end
        end
        
        % Input to fractional layer (if any)
        if fractional_layer > 0
            POC_frac_layer = TPOCin * dh * fractional_layer / hPOC * dt;
            fractional_layer_idx = n - num_full_layers;
            if fractional_layer_idx >= 1
                Ms_temp(3,fractional_layer_idx) = Ms_temp(3,fractional_layer_idx) + POC_frac_layer;
            end
        end
    end
    Ms=Ms_temp;
    %% Physical Uplift and Erosion Process
    % Calculate current layer thickness sequence
    dep_seq = zeros(n, 1);
    for j = 1:n
        dep_seq(j) = sum(Ms_temp(:,j) ./ den') / S0;
    end
    
    % Calculate erosion at top (layer n is the top layer)
    if sum(Ms_temp(:,n)) > 0
        rE = Ms_temp(:,n) / sum(Ms_temp(:,n));
        E = u * S0 * den(1) * (1 - WI0) * rE * dt;
    else
        rE = [1; 0; 0];
        E = u * S0 * den(1) * (1 - WI0) * rE * dt;
    end
    
    % Check if erosion thickness exceeds top layer thickness
    h_E = sum(E ./ den') / S0;
    if h_E > dep_seq(n)
        fprintf('Warning: Erosion thickness (%.6f m) exceeds top layer thickness (%.6f m) at t = %.0f years\n', ...
                h_E, dep_seq(n), t);
        h_E = min(h_E, dep_seq(n));
    end
    
    % Apply erosion to top layer and uplift to bottom layer
    dep_seq_E = dep_seq;
    Ms_E = Ms_temp;
    
    if h_E > 0
        % Update top layer after erosion
        Ms_E(:,n) = Ms_temp(:,n) - E;
        dep_seq_E(n) = sum(Ms_E(:,n) ./ den') / S0;
        Ms_E(:,n) = max(0, Ms_E(:,n));
    end
    
    % Add uplift at bottom (layer 1 is the bottom layer)
    Rockin = u * S0 * den(1) * dt;
    Ms_E(1,1) = Ms_temp(1,1) + Rockin;
    dep_seq_E(1) = sum(Ms_E(:,1) ./ den') / S0;
    
    % Update variables
    Ms = Ms_E;
    dep_seq = dep_seq_E;
    
    % Calculate new total thickness
    h_new = sum(dep_seq);
    
    %% Update Profile Geometry with Grid Remapping
    % Always remap to new grid
    % Calculate old depth coordinates
    z_old_bottom = zeros(n, 1);
    z_old_top = zeros(n, 1);
    for j = 1:n
        z_old_top(j) = sum(dep_seq(1:j-1));
        z_old_bottom(j) = sum(dep_seq(1:j));
    end
    
    % Calculate solid mass line density
    dMs = zeros(3, n);
    for j = 1:n
        if dep_seq(j) > 0
            dMs(:,j) = Ms(:,j) / dep_seq(j);
        else
            dMs(:,j) = 0;
        end
    end
    
    % Calculate dissolved mass line density
    Cd_line = zeros(n, 1);
    for j = 1:n
        if dep_seq(j) > 0
            Cd_line(j) = Cd(j) * layer_poro(j);
        else
            fprintf('Warning: dissolved matters overcosted');
            Cd_line(j) = 0;
        end
    end
    
    % Create new uniform grid
    dh_new = h_new / n;
    z_new_top = zeros(n, 1);
    z_new_bottom = zeros(n, 1);
    for i = 1:n
        z_new_top(i) = (i-1) * dh_new;
        z_new_bottom(i) = i * dh_new;
    end
    
    % Calculate overlap thickness between old and new layers
    h_overlap = zeros(n, n);
    for i = 1:n
        for j = 1:n
            overlap_top = max(z_new_top(i), z_old_top(j));
            overlap_bottom = min(z_new_bottom(i), z_old_bottom(j));
            h_overlap(i,j) = max(0, overlap_bottom - overlap_top);
        end
    end
    
    % Calculate new solid masses
    Ms_new = zeros(3, n);
    for i = 1:n
        for k = 1:3
            for j = 1:n
                if h_overlap(i,j) > 0
                    Ms_new(k,i) = Ms_new(k,i) + dMs(k,j) * h_overlap(i,j);
                end
            end
        end
    end
    
    % Calculate new dissolved line density
    Cd_line_new = zeros(n, 1);
    for i = 1:n
        for j = 1:n
            if h_overlap(i,j) > 0
                Cd_line_new(i) = Cd_line_new(i) + Cd_line(j) * h_overlap(i,j);
            end
        end
    end
    
    % Calculate new porosity and concentration
    Cd_new = zeros(n, 1);
    layer_poro_new = zeros(n, 1);
    dep_seq_new = zeros(n, 1);
    
    for i = 1:n
        total_volume = sum(Ms_new(:,i) ./ den');
        if total_volume > 0
            layer_poro_new(i) = sum((Ms_new(:,i) ./ den') .* poro') / total_volume;
        else
            layer_poro_new(i) = 0;
        end
        
        if layer_poro_new(i) > 0 && dh_new > 0
            Cd_new(i) = Cd_line_new(i) / (layer_poro_new(i) * dh_new);
        else
            Cd_new(i) = 0;
        end
        
        dep_seq_new(i) = sum(Ms_new(:,i) ./ den') / S0;
    end
    
    % Verify volume conservation after remapping
    h_newtest = sum(dep_seq_new);
    volume_change = h_newtest - h_new;
    if abs(volume_change) > 1e-10
        fprintf('Warning: Remapping caused volume change = %.2e m at t = %.0f years\n', volume_change, t);
    end
    
    % Update variables
    Ms = Ms_new;
    Cd = Cd_new;
    layer_poro = layer_poro_new;
    h = h_new;
    dh = dh_new;
    
    % Final thickness verification
    final_thickness_check = 0;
    for j = 1:n
        final_thickness_check = final_thickness_check + sum(Ms(:,j) ./ den') / S0;
    end
    if abs(final_thickness_check - h_new) > 1e-10
        fprintf('Warning: Final thickness mismatch = %.2e m at t = %.0f years\n', abs(final_thickness_check - h_new), t);
    end
    
    %% Calculate Volume Changes for Monitoring
    % Recalculate erosion for volume change calculations
    if sum(Ms(:,n)) > 0
        rE = Ms(:,n) / sum(Ms(:,n));
        E = u * S0 * den(1) * (1 - WI0) * rE * dt;
    else
        rE = [1; 0; 0];
        E = u * S0 * den(1) * (1 - WI0) * rE * dt;
    end
    
    dVER = -E(1)/den(1)*dt - E(2)/den(2)*dt; % Rock erosion volume change
    dVEC = -E(3)/den(3)*dt; % Organic matter erosion volume change
    dVWR = sum(Reac_solid(2,:))/den(2)*dt + sum(Reac_solid(1,:))/den(1)*dt; % Weathering volume change
    dVOX = TPOCin/den(3)*dt+sum(Reac_solid(3,:))/den(3)*dt; % Organic volume change
    dhV = u * dt + dVER + dVEC + dVWR + dVOX; % Total volume change
    
    % Check thickness error
    thickness_error = dhV - (h - h_prev);
    if abs(thickness_error) > 1e-10
        fprintf('Warning: Thickness error = %.2e m at t = %.0f years\n', thickness_error, t);
    end
    
    %% Check Convergence
    % Calculate relative changes
    Ms_rel_change = max(abs(Ms - Ms_prev) ./ (abs(Ms_prev) + eps), [], 'all');
    Cd_rel_change = max(abs(Cd - Cd_prev) ./ (abs(Cd_prev) + eps));
    dhV_rate = dhV / dt;
    
    max_rel_change = max([Ms_rel_change, Cd_rel_change]);
    
    if dhV_rate < 1e-6 && max_rel_change < tolerance
        converged = true;
        fprintf('Convergence achieved at t = %.0f years\n', t);
    end
    
    %% Record Data Every 1000 Years
    if mod(iteration, 1000) == 0 || converged
        time_record(end+1) = t;
        h_record(end+1) = h;
        
        % Store profile composition
        profile_record(:,:,end+1) = Ms;
        
                % Store volume changes and convergence metrics
        volume_changes(end+1, :) = [t, dVER, dVEC, dVWR, dVOX, dhV];
        convergence_metrics(end+1, :) = [t, Ms_rel_change, Cd_rel_change, dhV_rate, max_rel_change];
        
        fprintf('%8.0f\t%12.3f\t%15.2e\n', t, h, max_rel_change);
    end
    
    %% Display Progress
    if mod(iteration, 10000) == 0
        fprintf('Iteration: %d, Time: %.0f years, Thickness: %.3f m, Max Change: %.2e\n', ...
                iteration, t, h, max_rel_change);
    end
end

%% Final Calculations and Output
fprintf('\n=== Simulation Completed ===\n');
fprintf('Total simulation time: %.0f years\n', t);
fprintf('Final profile thickness: %.3f m\n', h);
fprintf('Number of iterations: %d\n', iteration);
fprintf('Convergence status: %s\n', converged, 'Achieved' : 'Not achieved');

% Calculate final erosion rates
if sum(Ms(:,n)) > 0
    rE_final = Ms(:,n) / sum(Ms(:,n));
    E_final = u * S0 * den(1) * (1 - WI0) * rE_final * dt;
else
    rE_final = [1; 0; 0];
    E_final = u * S0 * den(1) * (1 - WI0) * rE_final * dt;
end

Rockin_final = u * S0 * den(1) * dt;

% Calculate organic matter input for final ratio
if hPOC > h
    POCin_final = TPOCin * dt; % Total input
else
    POCin_final = TPOCin * dt; % Total input is TPOCin * dt
end

% Final metrics
WI_final = (E_final(1) + E_final(2)) / Rockin_final;
org_erosion_ratio = E_final(3) / sum(E_final);
if POCin_final > 0
    org_input_ratio = E_final(3) / POCin_final;
else
    org_input_ratio = 0;
end

%% Output Final Results
fprintf('\n=== Final Results ===\n');
fprintf('Final weathering intensity: %.4f (target: %.1f)\n', WI_final, WI0);
fprintf('Organic matter erosion ratio: %.4f\n', org_erosion_ratio);
fprintf('Organic matter input utilization: %.4f\n', org_input_ratio);

fprintf('\n=== Final Erosion Fluxes ===\n');
fprintf('Rock erosion: %.2f g/year\n', E_final(1)/dt);
fprintf('Weathered product erosion: %.2f g/year\n', E_final(2)/dt);
fprintf('Organic matter erosion: %.2f g/year\n', E_final(3)/dt);

fprintf('\n=== Volume Change Rates ===\n');
fprintf('Uplift rate: %.2e m/year\n', u);
fprintf('Rock erosion volume change: %.2e m/year\n', dVER/dt);
fprintf('Organic erosion volume change: %.2e m/year\n', dVEC/dt);
fprintf('Weathering volume change: %.2e m/year\n', dVWR/dt);
fprintf('Oxidation volume change: %.2e m/year\n', dVOX/dt);
fprintf('Total volume change: %.2e m/year\n', dhV/dt);

%% Plot Results
if ~isempty(time_record)
    % Create depth array for plotting
    z_final = linspace(0, h, n)';
    
    % Calculate final composition fractions
    mass_fractions = zeros(n, 3);
    for j = 1:n
        total_mass = sum(Ms(:,j));
        if total_mass > 0
            mass_fractions(j,:) = Ms(:,j)' / total_mass;
        end
    end
    
    % Main results figure
    figure('Position', [100, 100, 1500, 1000]);
    
    % 1. Final composition profile
    subplot(3, 4, 1);
    plot(mass_fractions(:,1), z_final, 'r-', 'LineWidth', 2);
    hold on;
    plot(mass_fractions(:,2), z_final, 'b-', 'LineWidth', 2);
    plot(mass_fractions(:,3), z_final, 'g-', 'LineWidth', 2);
    xlabel('Mass Fraction');
    ylabel('Depth [m]');
    legend('Rock', 'Weathered', 'Organic', 'Location', 'best');
    title('Final Composition Profile');
    grid on;
    set(gca, 'YDir', 'reverse');
    
    % 2. CO2 concentration profile
    subplot(3, 4, 2);
    plot(Cd, z_final, 'k-', 'LineWidth', 2);
    xlabel('CO_2 Concentration [mol/m?]');
    ylabel('Depth [m]');
    title('CO_2 Concentration Profile');
    grid on;
    set(gca, 'YDir', 'reverse');
    
    % 3. Porosity profile
    subplot(3, 4, 3);
    plot(layer_poro, z_final, 'm-', 'LineWidth', 2);
    xlabel('Porosity');
    ylabel('Depth [m]');
    title('Porosity Profile');
    grid on;
    set(gca, 'YDir', 'reverse');
    
    % 4. Mass distribution
    subplot(3, 4, 4);
    semilogy(Ms(1,:)', z_final, 'r-', 'LineWidth', 2);
    hold on;
    semilogy(Ms(2,:)', z_final, 'b-', 'LineWidth', 2);
    semilogy(Ms(3,:)', z_final, 'g-', 'LineWidth', 2);
    xlabel('Mass per Layer [g]');
    ylabel('Depth [m]');
    legend('Rock', 'Weathered', 'Organic', 'Location', 'best');
    title('Mass Distribution');
    grid on;
    set(gca, 'YDir', 'reverse');
    
    % 5. Thickness evolution
    subplot(3, 4, 5);
    plot(time_record/1000, h_record, 'b-', 'LineWidth', 2);
    xlabel('Time [kyr]');
    ylabel('Profile Thickness [m]');
    title('Thickness Evolution');
    grid on;
    
    % 6. Volume change rates evolution
    subplot(3, 4, 6);
    if size(volume_changes, 1) > 1
        time_vol = volume_changes(2:end,1)/1000;
        plot(time_vol, volume_changes(2:end,2)/dt, 'r-', 'LineWidth', 1);
        hold on;
        plot(time_vol, volume_changes(2:end,3)/dt, 'g-', 'LineWidth', 1);
        plot(time_vol, volume_changes(2:end,4)/dt, 'b-', 'LineWidth', 1);
        plot(time_vol, volume_changes(2:end,5)/dt, 'm-', 'LineWidth', 1);
        plot(time_vol, volume_changes(2:end,6)/dt, 'k-', 'LineWidth', 2);
        xlabel('Time [kyr]');
        ylabel('Volume Change Rate [m/year]');
        legend('dVER', 'dVEC', 'dVWR', 'dVOX', 'dhV', 'Location', 'best');
        title('Volume Change Rates');
        grid on;
    end
    
    % 7. Convergence metrics
    subplot(3, 4, 7);
    if ~isempty(convergence_metrics)
        semilogy(convergence_metrics(:,1)/1000, convergence_metrics(:,2), 'ro-', 'MarkerSize', 3);
        hold on;
        semilogy(convergence_metrics(:,1)/1000, convergence_metrics(:,3), 'bs-', 'MarkerSize', 3);
        semilogy(convergence_metrics(:,1)/1000, convergence_metrics(:,5), 'k-', 'LineWidth', 2);
        xlabel('Time [kyr]');
        ylabel('Relative Change');
        legend('Ms', 'Cd', 'Max', 'Location', 'best');
        title('Convergence Metrics');
        grid on;
    end
    
    % 8. Volume change rate convergence
    subplot(3, 4, 8);
    if ~isempty(convergence_metrics)
        semilogy(convergence_metrics(:,1)/1000, abs(convergence_metrics(:,4)), 'g-', 'LineWidth', 2);
        xlabel('Time [kyr]');
        ylabel('|dhV/dt| [m/year]');
        title('Volume Change Rate Convergence');
        grid on;
        yline(1e-6, 'r--', 'Convergence Threshold', 'LineWidth', 2);
    end
    
    % 9. Erosion fluxes bar chart
    subplot(3, 4, 9);
    erosion_fluxes = E_final / dt;
    bar(erosion_fluxes, 'FaceColor', [0.7 0.7 0.7]);
    set(gca, 'XTickLabel', {'Rock', 'Weathered', 'Organic'});
    ylabel('Erosion Flux [g/year]');
    title('Final Erosion Fluxes');
    grid on;
    
    % 10. Erosion composition pie chart
    subplot(3, 4, 10);
    if sum(erosion_fluxes) > 0
        pie(erosion_fluxes, {'Rock', 'Weathered', 'Organic'});
        title('Erosion Composition');
    end
    
    % 11. Key metrics display
    subplot(3, 4, 11);
    metrics = [WI_final, org_erosion_ratio, org_input_ratio];
    bar(metrics, 'FaceColor', [0.5 0.8 0.9]);
    set(gca, 'XTickLabel', {'WI', 'Org Erosion', 'Org Input Util'});
    ylabel('Ratio');
    title('Key Performance Metrics');
    grid on;
    
    % 12. Text summary
    subplot(3, 4, 12);
    axis off;
    text(0.1, 0.9, sprintf('Final Thickness: %.3f m', h), 'FontSize', 10);
    text(0.1, 0.8, sprintf('Simulation Time: %.0f years', t), 'FontSize', 10);
    text(0.1, 0.7, sprintf('Weathering Intensity: %.4f', WI_final), 'FontSize', 10);
    text(0.1, 0.6, sprintf('Org Erosion Ratio: %.4f', org_erosion_ratio), 'FontSize', 10);
    text(0.1, 0.5, sprintf('Org Input Utilization: %.4f', org_input_ratio), 'FontSize', 10);
    text(0.1, 0.4, sprintf('Final Convergence: %.2e', max_rel_change), 'FontSize', 10);
    text(0.1, 0.3, sprintf('Volume Change Rate: %.2e m/yr', dhV_rate), 'FontSize', 10);
    
    sgtitle('Weathering Profile Simulation - Comprehensive Results');
    
    %% Additional Detailed Plots
    % Evolution of composition profiles
    if size(profile_record, 3) > 3
        figure('Position', [100, 100, 1200, 800]);
        
        % Select time points to display
        time_indices = round(linspace(1, size(profile_record, 3), 4));
        time_indices(end) = size(profile_record, 3);
        
        for i = 1:4
            idx = time_indices(i);
            t_display = time_record(idx);
            h_display = h_record(idx);
            z_display = linspace(0, h_display, n)';
            
            subplot(2, 4, i);
            Ms_temp = profile_record(:,:,idx);
            mass_frac_temp = zeros(n, 3);
            for j = 1:n
                total_mass = sum(Ms_temp(:,j));
                if total_mass > 0
                    mass_frac_temp(j,:) = Ms_temp(:,j)' / total_mass;
                end
            end
            
            plot(mass_frac_temp(:,1), z_display, 'r-', 'LineWidth', 2);
            hold on;
            plot(mass_frac_temp(:,2), z_display, 'b-', 'LineWidth', 2);
            plot(mass_frac_temp(:,3), z_display, 'g-', 'LineWidth', 2);
            xlabel('Mass Fraction');
            ylabel('Depth [m]');
            title(sprintf('t = %.0f years', t_display));
            grid on;
            set(gca, 'YDir', 'reverse');
            
            if i == 1
                legend('Rock', 'Weathered', 'Organic', 'Location', 'best');
            end
            
            subplot(2, 4, i+4);
            % Calculate porosity for this time
            poro_temp = zeros(n, 1);
            for j = 1:n
                total_volume = sum(Ms_temp(:,j) ./ den');
                if total_volume > 0
                    poro_temp(j) = sum((Ms_temp(:,j) ./ den') .* poro') / total_volume;
                end
            end
            
            plot(poro_temp, z_display, 'm-', 'LineWidth', 2);
            xlabel('Porosity');
            ylabel('Depth [m]');
            title(sprintf('Porosity at t = %.0f years', t_display));
            grid on;
            set(gca, 'YDir', 'reverse');
        end
        
        sgtitle('Evolution of Composition and Porosity Profiles');
    end
    
    % Cumulative mass distribution
    figure('Position', [100, 100, 1000, 400]);
    
    subplot(1, 2, 1);
    cum_rock = cumsum(Ms(1,:), 'reverse');
    cum_weathered = cumsum(Ms(2,:), 'reverse');
    cum_organic = cumsum(Ms(3,:), 'reverse');
    
    plot(cum_rock/1000, z_final, 'r-', 'LineWidth', 2);
    hold on;
    plot(cum_weathered/1000, z_final, 'b-', 'LineWidth', 2);
    plot(cum_organic/1000, z_final, 'g-', 'LineWidth', 2);
    xlabel('Cumulative Mass [kg/m?]');
    ylabel('Depth [m]');
    legend('Rock', 'Weathered', 'Organic', 'Location', 'best');
    title('Cumulative Mass Distribution');
    grid on;
    set(gca, 'YDir', 'reverse');
    
    subplot(1, 2, 2);
    % Reaction rates profile (calculate from final state)
    reac_rates = zeros(n, 3);
    rate_factor = exp(-Ea/R * (1/T - 1/T0));
    
    for j = 1:n
        % Weathering reaction rate
        if Ms(1,j) > 0 && Cd(j) > 0
            Sa_rock = Ms(1,j) / den(1) * (1 - poro(1)) / (d/3);
            reac_rates(j,1) = -rw * Sa_rock * (Cd(j)/Cd_atm)^a * rate_factor / (Ms(1,j) + eps);
            reac_rates(j,2) = -reac_rates(j,1) * m_ratio / (Ms(2,j) + eps);
        end
        
        % Organic oxidation rate
        if Ms(3,j) > 0
            reac_rates(j,3) = -rox * Ms(3,j) * rate_factor / (Ms(3,j) + eps);
        end
    end
    
    semilogy(abs(reac_rates(:,1)), z_final, 'r-', 'LineWidth', 2);
    hold on;
    semilogy(abs(reac_rates(:,2)), z_final, 'b-', 'LineWidth', 2);
    semilogy(abs(reac_rates(:,3)), z_final, 'g-', 'LineWidth', 2);
    xlabel('Normalized Reaction Rate [1/year]');
    ylabel('Depth [m]');
    legend('Rock weathering', 'Weathered production', 'Organic oxidation', 'Location', 'best');
    title('Normalized Reaction Rates');
    grid on;
    set(gca, 'YDir', 'reverse');
    
    sgtitle('Additional Analysis');
    
    fprintf('\nPlotting completed. Generated comprehensive results figures.\n');
else
    fprintf('No recorded data available for plotting.\n');
end

%% Final Summary and Data Export
fprintf('\n=== Simulation Summary ===\n');
fprintf('Initial thickness: 1.000 m\n');
fprintf('Final thickness: %.3f m\n', h);
fprintf('Thickness change: %.3f m\n', h - 1.0);
fprintf('Convergence achieved: %s\n', converged, 'Yes' : 'No');
fprintf('Total iterations: %d\n', iteration);
fprintf('Final weathering intensity: %.4f (target: %.1f)\n', WI_final, WI0);
fprintf('Organic matter cycling efficiency: %.4f\n', org_input_ratio);


    