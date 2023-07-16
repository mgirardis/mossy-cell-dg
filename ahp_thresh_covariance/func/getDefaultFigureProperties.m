function s = getDefaultFigureProperties()
    s.pLines = {'--'};
    s.pSymbols = [ 'o', '^', 'v', 's', '>', '<', 'p', 'h', 'x', '+', '*' ];
    s.pColors = [ [112,179,250;... % azul
                   210,45,45;... % vermelho
                   229,173,68;... % laranja
                   26,92,186;... % azul escuro
                   186,26,26;... % vermelho escuro
                   186,87,26;... % laranja escuro (marrom)
                   99,199,113;... % verde
                   194,129,201;... % roxo
                   210,217,78]./255;... % amarelo
                   lines(7) ];

    s.pPointSize = 4;
    s.pPointFaceColor = 'w';
    s.pLineColor = 'k';
    s.pLineWidth = 1;

    s.pErrSymbol = '';
    s.pErrLineSpec = '-';
    s.pErrLineWidth = 0.5;
    s.pErrLineColor = 'auto';

    s.wFig = 800;
    s.hFig = 300;
    ss = get(0, 'ScreenSize');
    mp = get(0, 'MonitorPositions');
    if size(mp,1) == 2
        ss = mp(2,:);
    end
    s.x0Fig = ss(1)+randi(floor([100,(ss(3)-s.wFig-100)]), 1, 1);
    s.y0Fig = ss(2)+randi(floor([100,(ss(4)-s.hFig-100)]), 1, 1);
    s.margin = 0.15;
    s.rtMarginPer = 0.2;
    s.hwRatio = s.hFig/s.wFig;
    s.lMargin = s.margin*s.hwRatio;
    s.bMargin = s.margin;
    s.hAx = 1 - s.bMargin - s.rtMarginPer*s.bMargin;
    s.wAx = (1 - 2*s.lMargin) / 2 - s.rtMarginPer*s.lMargin*s.hwRatio;
end
