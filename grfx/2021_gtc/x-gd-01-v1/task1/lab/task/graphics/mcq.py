##Basic mcq

from ipywidgets import widgets, Layout, Box, GridspecLayout

def create_multipleChoice_widget(description, options, correct_answer, tip):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False,
        indent = False,
        align = 'center',
    )
    
    description_out = widgets.Output(layout=Layout(width='auto'))
    
    with description_out:
        print(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "correct" + '\x1b[0m' +"\n"
        else:
            s = '\x1b[5;30;41m' + "try again" + '\x1b[0m' +"\n"
        with feedback_out:
            feedback_out.clear_output()
            print(s)
        return
    
    check = widgets.Button(description="check")
    check.on_click(check_selection)
    
    tip_out = widgets.Output()
    
    def tip_selection(b):
        with tip_out:
            print(tip)
            
        with feedback_out:
            feedback_out.clear_output()
            print(tip)
    
    tipbutton = widgets.Button(description="tip")
    tipbutton.on_click(tip_selection)
    
    return widgets.VBox([description_out, 
                         alternativ, 
                         widgets.HBox([tipbutton, check]), feedback_out], 
                        layout=Layout(display='flex',
                                     flex_flow='column',
                                     align_items='stretch',
                                     width='auto')) 