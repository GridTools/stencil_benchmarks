#include "knl/knl_basic_multifield_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template class basic_multifield_variant<flat, float>;
        template class basic_multifield_variant<flat, double>;
        template class basic_multifield_variant<cache, float>;
        template class basic_multifield_variant<cache, double>;

    } // namespace knl

} // namespace platform
