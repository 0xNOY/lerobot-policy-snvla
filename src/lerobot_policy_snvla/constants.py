OBS_LANGUAGE = "observation.language"
OBS_LANGUAGE_TOKEN_AR_MASK = OBS_LANGUAGE + ".ar_mask"
OBS_LANGUAGE_TOKEN_LOSS_MASK = OBS_LANGUAGE + ".loss_mask"
OBS_LANGUAGE_MODE_MASK = OBS_LANGUAGE + ".mode_mask"

CURRENT_NARRATION = "current_narration"
PREVIOUS_NARRATIONS = "previous_narrations"

STATE_DROPOUT_MASK = "state_dropout_mask"
TRAINING_EPOCH = "training_epoch"
NARRATION_TARGET_MASK = "narration_target_mask"

# Deprecated compatibility constants; migrate consumers and remove these aliases in Tasks 2/3.
DIFFUSION_LOSS_MASK = "diffusion_loss_mask"
STATE_RANDOMIZED_TEXT_ONLY_MASK = "state_randomized_text_only_mask"

COMPLEMENTARY_DATA = "complementary_data"
