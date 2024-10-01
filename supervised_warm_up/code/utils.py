
class Action:
    retrieve = 'retrieve'
    reason = 'reason'
    end = 'end'

    @classmethod
    def linearize_action(cls, action):
        action_str = ""

        if action['type'] == Action.retrieve:
            action_str += f"{action['type']}: {action['query_id']}"

        elif action['type'] == Action.reason:
            action_str += f"{action['type']}: "
            action_str += " & ".join(sorted(action['step']['pre_id']))
            if 'con_sent' in action['step']:
                action_str += f" -> {action['step']['con_sent']}"

        elif action['type'] == Action.end:
            action_str += f"{action['type']}"
            if 'is_proved' in action:
                action_str += f": {action['is_proved']}"

        else:
            raise NotImplementedError

        return action_str

    @classmethod
    def parse_action(cls, action_str):
    
        if ':' in action_str:
            action_type, paras_str = action_str.split(':', maxsplit=1)
            action_type = action_type.strip()
        else:
            action_type = action_str.strip()
            paras_str = None

            
        if action_type == Action.retrieve:
            action = {
                'type': action_type,
                'query_id': paras_str.strip(),
            }
        
        elif action_type == Action.reason:
            if '->' not in paras_str:
                pre_id = [p.strip() for p in paras_str.split('&')]
                action = {
                    'type': action_type,
                    'step': {
                        'pre_id': pre_id,
                    },
                    'use_module': True,
                }
            else:
                pre_id_str, con_sent = paras_str.split('->', maxsplit=1)
                pre_id = [p.strip() for p in pre_id_str.split('&')]
                action = {
                    'type': action_type,
                    'step': {
                        'pre_id': pre_id,
                        'con_sent': con_sent.strip(),
                    },
                    'use_module': False,
                }

        elif action_type == Action.end:
            if paras_str is None:
                action = {
                    'type': action_type,
                }
            else:
                action = {
                    'type': action_type,
                    'is_proved': paras_str.strip(),
                }  
        else:
            action = None

        return action
    

def chunk(it, n):
    c = []
    for x in it:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c