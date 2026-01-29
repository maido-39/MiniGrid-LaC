######################################################
#                                                    #
#                       USER                         #
#                    INTERACTION                     #
#                                                    #
######################################################


"""
User Interaction Module

This module handles user interactions such as receiving input from the user.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


# None




######################################################
#                                                    #
#                        CLASS                       #
#                                                    #
######################################################


class UserInteraction:
    """User Interaction Class"""
    
    def get_input(self, prompt: str = "> ") -> str:
        """Receive user input"""
        
        return input(prompt).strip()



