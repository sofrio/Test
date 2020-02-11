function updateFig(fh,main,ustep,hstep)

if isempty(fh)
    return;
end
figure(fh);
if exist('main','var') && ~isempty(main)
    if ~isempty(main{1})
        set(findobj(fh,'Tag','Label_MainStep'),'String',['Main ',num2str(main{1})]);
    end
    if ~isempty(main{2})
        subplot(224); %mesh(main{2}); 
        dispIm(main{2});
    end
end
if exist('ustep','var') && ~isempty(ustep)
    if ~isempty(ustep{1})
        set(findobj(fh,'Tag','Label_UStep'),'String',['Ustep ',num2str(ustep{1})]);
    end
    if ~isempty(ustep{2})
        subplot(221); dispIm(ustep{2});
    end
    if ~isempty(ustep{3})
        subplot(222); dispIm(ustep{3});
    end
end
if exist('hstep','var') && ~isempty(hstep)
    if ~isempty(hstep{1})
        set(findobj(fh,'Tag','Label_HStep'),'String',['Hstep ',num2str(hstep{1})]);
    end
    if ~isempty(hstep{2})
        subplot(223); dispIm(hstep{2});
    end
end
drawnow;

