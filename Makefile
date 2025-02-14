# .PHONY: lib, pybind, clean, format, all

# all: lib


# lib:
# 	@mkdir -p build
# 	@cd build; cmake ..
# 	@cd build; $(MAKE)

# format:
# 	python3 -m black .
# 	clang-format -i src/*.cc src/*.cu

# clean:
# 	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so

.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@if not exist build mkdir build
	@cd build && cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
	@cmake --build build --config Release

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	if exist build rmdir /s /q build
	if exist python\needle\backend_ndarray\Release\*.pyd del python\needle\backend_ndarray\Release\*.pyd