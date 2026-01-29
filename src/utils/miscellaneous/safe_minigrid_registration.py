######################################################
#                                                    #
#                     SAFE MINIGRID                  #
#                     REGISTRATION                   #
#                                                    #
######################################################


"""
This program aims to safely register a custom MiniGrid environment.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


from gymnasium.envs.registration import registry, register

from utils.miscellaneous.global_variables import ENV_ID




######################################################
#                                                    #
#                      FUNCTION                      #
#                                                    #
######################################################


def safe_minigrid_reg():
    """
    Safely register a custom MiniGrid environment if not already registered.
    """
    
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="my_package.my_env:MyEnv",
        )