template <unsigned N>
struct factorial {
	static constexpr unsigned value = N * factorial<N - 1>::value;
};

// Usage examples:
// factorial<0>::value would yield 1;
// factorial<4>::value would yield 24.