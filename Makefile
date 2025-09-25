# Makefile to package selected files into a zip (placed at repo root)

ZIP_NAME := supplement.zip
EXTS     := py md txt pt pth npy yaml

# Build the find-expression for included extensions
FIND_INCLUDES := $(foreach e,$(EXTS),-name '*.$(e)' -o) -false

.PHONY: all clean

all: $(ZIP_NAME)

$(ZIP_NAME):
	@rm -f $@
	@printf 'Creating %s ...\n' $@
	@find . -type f \( $(FIND_INCLUDES) \) \
		-not -path '*/.*' \
		-not -name 'Makefile' \
		-print \
	| LC_ALL=C sort \
	| zip -q -@ $@

clean:
	@rm -f $(ZIP_NAME)
