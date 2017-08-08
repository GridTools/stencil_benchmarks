#pragma once

#include <functional>

namespace platform {

    template < class... Platforms >
    struct platform_list;

    template <>
    struct platform_list<> {
        static constexpr bool empty = true;

        template < class F, class... Args >
        static void loop(Args &&...) {}
    };

    template < class Platform, class... Platforms >
    struct platform_list< Platform, Platforms... > {
        static constexpr bool empty = false;

        template < class F, class... Args >
        static void loop(Args &&... args) {
            F::template execute< Platform >(std::forward< Args >(args)...);
            platform_list< Platforms... >::template loop< F >(std::forward< Args >(args)...);
        }
    };

    template < class PlatformList1, class PlatformList2 >
    struct merge_platform_list;

    template < class... Platforms1, class... Platforms2 >
    struct merge_platform_list< platform_list< Platforms1... >, platform_list< Platforms2... > > {
        using type = platform_list< Platforms1..., Platforms2... >;
    };

} // namespace platform
