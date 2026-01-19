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


import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg




######################################################
#                                                    #
#                   INITIALIZATION                   #
#                                                    #
######################################################


# MiniGrid Environment Safe Registration
safe_minigrid_reg()




######################################################
#                                                    #
#                      FUNCTION                      #
#                                                    #
######################################################


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            tfu.cprint("-- Usage Instructions --", tfu.LIGHT_GREEN, True)
            tfu.cprint("python minigrid_lac.py <json_map_path>", tfu.LIGHT_RED, italic=True, indent=8)
            tfu.cprint("Example: python minigrid_lac.py scenario135_example_map.json", tfu.LIGHT_BLACK, italic=True)
            return
        else:
            json_map_path = sys.argv[1]
    else:
        # Specify the path to the JSON map file as a command-line argument
        json_map_path = "scenario135_example_map.json"
    
    try:
        experiment = ScenarioExperiment(json_map_path = json_map_path)
        experiment.run()
    except KeyboardInterrupt:
        tfu.cprint("\n\nTerminated by the user!", tfu.LIGHT_BLUE, True)
    except Exception as e:
        tfu.cprint(f"\n\nError occurred: {e}", tfu.LIGHT_RED, True)
        import traceback
        traceback.print_exc()




######################################################
#                                                    #
#                        MAIN                        #
#                                                    #
######################################################


if __name__ == "__main__":
    main()