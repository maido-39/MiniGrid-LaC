######################################################
#                                                    #
#                      MiniGrid-LaC                  #
#                        PROGRAM                     #
#                                                    #
######################################################


"""Main Program"""




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
    """
    
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="my_package.my_env:MyEnv",
        )