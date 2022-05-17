template <unsigned N>
struct factorial {
	static constexpr unsigned value = N * factorial<N - 1>::value;
};

template <>
struct factorial<0> {
	static constexpr unsigned value = 1;
};

// Usage examples:
// factorial<0>::value would yield 1;
// factorial<4>::value would yield 24.