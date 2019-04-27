ts() {
    gawk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }'
}
