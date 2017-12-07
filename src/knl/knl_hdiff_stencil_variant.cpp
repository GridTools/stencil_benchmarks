#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template class hdiff_stencil_variant<flat, float>;
        template class hdiff_stencil_variant<flat, double>;
        template class hdiff_stencil_variant<cache, float>;
        template class hdiff_stencil_variant<cache, double>;

    } // namespace knl

} // namespace platform
