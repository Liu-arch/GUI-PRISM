ACTIONS = ('Click','LongPress','Write','Select','Scroll','Swipe','KeyboardPress',
           'OpenApp','NavigateBack','NavigateHome','Wait','Terminate','Answer')
TEXT_TYPES = frozenset(('Write','Select','OpenApp','Answer'))
MANDATORY_POINT = frozenset(('Click','LongPress','Select'))
OPTIONAL_POINT = frozenset(('Write','KeyboardPress','Scroll'))
