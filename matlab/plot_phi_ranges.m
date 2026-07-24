function plot_phi_ranges()
% Read CSV
tbl = readtable('phase_offsets.csv','PreserveVariableNames',true);
% Expect variables: Min, Max, Loading, K Spacing, Direction,
%                   Min Failure Mode, Max Failure Mode
vars = lower(tbl.Properties.VariableNames);
getcol = @(name) tbl{:, strcmpi(vars, lower(name))};
phase_min = getcol('min');
phase_max = getcol('max');
loading   = getcol('loading');
k         = getcol('spring constant (n/mm)');
spacing   = getcol('spacing');
direction = getcol('direction');
min_mode  = getcol('min failure mode');
max_mode  = getcol('max failure mode');

% Prepare figure
fig = figure;
% Reserve room on the right for the legend BEFORE calling breakyaxis.
% breakyaxis resets the axes' Position to fill the whole figure, which
% is what was pushing the plot on top of an 'eastoutside' legend.
ax = axes('Parent',fig,'Position',[0.10 0.11 0.62 0.82]);
hold(ax,'on');

% Colors for spacing
colors = containers.Map([10, 20, 30],{[0.5 0 0], [0.5 0 0.5],[0 0.5 0]}); % dark red, purple, green
n = numel(k);
for i=1:n
    x = [phase_min(i), phase_max(i)];
    y = [k(i), k(i)];
    if isKey(colors,spacing(i))
        c = colors(spacing(i));
    else
        c = [0 0 0];
    end
    % % Distinguish Forward vs Reverse direction by line style
    % if strcmpi(strtrim(direction{i}),'R')
    %     lineStyle = '--';
    % else
    lineStyle = '-';
    % end
    plot(ax,x,y,lineStyle,'Color',c,'LineWidth',1.5);
    % left marker
    switch lower(strtrim(min_mode{i}))
        case 'slip'
            plot(ax,phase_min(i),k(i),'o','MarkerFaceColor','none','MarkerEdgeColor',c,'MarkerSize',6);
        case 'squeeze'
            plot(ax,phase_min(i),k(i),'o','MarkerFaceColor',c,'MarkerEdgeColor',c,'MarkerSize',6);
        otherwise
            plot(ax,phase_min(i),k(i),'o','MarkerFaceColor','r','MarkerEdgeColor','r','MarkerSize',6);
    end
    % right marker
    switch lower(strtrim(max_mode{i}))
        case 'slip'
            plot(ax,phase_max(i),k(i),'o','MarkerFaceColor','none','MarkerEdgeColor',c,'MarkerSize',6);
        case 'squeeze'
            plot(ax,phase_max(i),k(i),'o','MarkerFaceColor',c,'MarkerEdgeColor',c,'MarkerSize',6);
        otherwise
            plot(ax,phase_max(i),k(i),'o','MarkerFaceColor','r','MarkerEdgeColor','r','MarkerSize',6);
    end
end

xlabel(ax,'\phi_{FB}'); ylabel(ax,'Spring Constant (N/mm)');
title(ax,'Phase offsets');
ylim(ax,[0.0 6001]);
breakyaxis(ax,[2.5, 5999]);
grid(ax,'on');

% --- Legend (built from dummy/proxy handles so entries aren't duplicated) ---
h = gobjects(1,6);
h(1) = plot(ax,nan,nan,'-','Color',[0.5 0 0],'LineWidth',1.5);     % spacing 10
h(2) = plot(ax,nan,nan,'-','Color',[0.5 0 0.5],'LineWidth',1.5);   % spacing 20
h(3) = plot(ax,nan,nan,'-','Color',[0 0.5 0],'LineWidth',1.5);     % spacing 30
% h(4) = plot(ax,nan,nan,'-','Color',[0 0 0],'LineWidth',1.5);       % direction F
% h(5) = plot(ax,nan,nan,'--','Color',[0 0 0],'LineWidth',1.5);      % direction R
h(4) = plot(ax,nan,nan,'o','MarkerFaceColor','none','MarkerEdgeColor',[1 1 1],'MarkerSize',6); % slip
h(5) = plot(ax,nan,nan,'o','MarkerFaceColor',[1 1 1],'MarkerEdgeColor',[1 1 1],'MarkerSize',6); % squeeze
h(6) = plot(ax,nan,nan,'o','MarkerFaceColor','r','MarkerEdgeColor','r','MarkerSize',6);         % unclear

lgd = legend(ax, h, {'Spacing = 10 mm','Spacing = 20 mm','Spacing = 30 mm', ...
           'Endpoint: Slip (open)','Endpoint: Squeeze (filled)','Endpoint: Unclear (red)'});
           % 'Direction: Forward (solid)','Direction: Reverse (dashed)', ...

% Place legend manually in the space reserved to the right of ax so that
% breakyaxis (called above) can't push the plot over it.
lgd.Units = 'normalized';
lgd.Position = [0.76 0.30 0.22 0.40];

hold(ax,'off');
end