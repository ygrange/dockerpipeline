.PHONY: all lofar-demopipeline lofar-wsclean lofar-base lofar-pipeline 

DOCKER_BUILD_COMMAND = docker build -t

define build
	$(DOCKER_BUILD_COMMAND) $(1) $(1)
endef

all: lofar-demopipeline
	echo done

lofar-demopipeline: lofar-wsclean
	$(call build, $@)

lofar-wsclean: lofar-pipeline
	$(call build, $@)

lofar-pipeline: lofar-base
	$(call build, $@)

lofar-base:
	$(call build, $@)

