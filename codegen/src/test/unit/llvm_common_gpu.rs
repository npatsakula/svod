use super::*;

#[test]
fn parse_special_axis_g_l_i() {
    assert_eq!(parse_special_axis("g0"), Some(('g', 0)));
    assert_eq!(parse_special_axis("gidx0"), Some(('g', 0)));
    assert_eq!(parse_special_axis("l1"), Some(('l', 1)));
    assert_eq!(parse_special_axis("lidx2"), Some(('l', 2)));
    assert_eq!(parse_special_axis("idx0"), Some(('i', 0)));
    assert_eq!(parse_special_axis("foo"), None);
    assert_eq!(parse_special_axis("g3"), None); // axis must be < 3
    assert_eq!(parse_special_axis("g"), None); // missing digit
}
