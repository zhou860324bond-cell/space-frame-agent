"""用户当前三杆节点的力矩和均布荷载不能遗漏或直接阻止子模型求解。"""

import copy

import numpy as np
import pytest

from agent import Session
from joint_loading import (abaqus_load_helper, collect_local_loads,
                           local_load_resultant, regional_load_forces)
from solid_joint import _script_text, prepare_joint_spec
import sections
from test_joint_transfer import _cantilever
from test_solid3d import _unit_tet
from test_solid_joint import _l_joint
from units import convert_model


def _loaded_t(units='N-m-Pa'):
    model = dict(units='N-m-Pa', materials=[dict(name='S', E=2e11, nu=.3)],
                 sections=[sections.i_section('H', .4, .2, .008, .013)],
                 nodes=[dict(id=i+1, x=x, y=y, z=0.) for i, (x,y) in
                        enumerate([(2.,0.), (0.,2.), (5.,2.), (2.,2.)])],
                 members=[dict(id=i+1, i=a, j=b, section='H', material='S', ref_vector=v)
                          for i, (a,b,v) in enumerate([(2,4,[0,1,0]), (4,3,[0,1,0]), (1,4,[-1,0,0])])],
                 supports=[dict(node=i, fix=[1]*6) for i in (1,2,3)],
                 load_cases=[dict(name='G', nodal_loads=[dict(node=4, load=[0,0,0,0,0,-5000])],
                                  member_loads=[dict(member=2, w=[0,-2000,0])],
                                  member_spans=[dict(member=1, kind='point', w1=[0,-10000,0], a=1.),
                                                dict(member=3, kind='point', w1=[10000,0,0], a=1.)])])
    if units != model['units']:
        model = convert_model(model, units)
    session = Session(model=model)
    assert session.solve_model().ok
    return session


@pytest.mark.parametrize('units', ['N-m-Pa', 'N-mm-MPa'])
def test_current_loaded_three_arm_node_rebuilds_moment_and_clipped_line_load(units):
    """复现当前节点4：不能因5kNm力矩和局部均布荷载拒绝求解，也不能重复跨中力。"""
    session = _loaded_t(units)
    spec = prepare_joint_spec(session, 4, loading_method='displacement', geometry_mode='transition')
    force, moment = local_load_resultant(spec.local_loads)
    length = spec.arms[1].length_mm
    assert force == pytest.approx([0., -2*length, 0.])
    assert moment == pytest.approx([0., 0., -5e6-length**2])
    assert sum(r['kind']=='point' for r in spec.local_loads) == 1
    cut_force = sum((np.asarray(a.cut_force_n) for a in spec.arms), np.zeros(3))
    cut_moment = sum((np.asarray(a.cut_moment_nmm)+np.cross(np.asarray(a.direction)*a.length_mm,a.cut_force_n)
                      for a in spec.arms), np.zeros(3))
    assert cut_force+force == pytest.approx(np.zeros(3), abs=1e-7)
    assert cut_moment+moment == pytest.approx(np.zeros(3), abs=1e-3)


@pytest.mark.parametrize('reverse', [False, True])
def test_clipped_ramp_load_keeps_first_moment_for_reversed_member(reverse):
    """防止反向杆或区间梯形的截取范围、强度插值和合力矩符号错误。"""
    session = _cantilever(reverse=reverse)
    case = session.model['load_cases'][0]
    a,b = (1.4,1.9) if reverse else (.1,.6)
    case['member_spans']=[dict(member=2,kind='partial_trapezoid',w1=[0,-100,0],w2=[0,-400,0],a=a,b=b)]
    assert session.solve_model().ok
    spec=prepare_joint_spec(session,2,loading_method='displacement')
    force,moment=local_load_resultant(spec.local_loads)
    end=min(.6,spec.arms[1].length_mm/1000)
    polynomial=np.polynomial.Polynomial([-460.,600.]) if reverse else np.polynomial.Polynomial([-40.,-600.])
    expected=polynomial.integ()(end)-polynomial.integ()(.1)
    first=(polynomial*np.polynomial.Polynomial([0.,1.])).integ()
    expected_m=(first(end)-first(.1))*1000
    assert force == pytest.approx([0,expected,0], abs=1e-7)
    assert moment == pytest.approx([0,0,expected_m], abs=1e-6)


def test_compiled_point_load_and_combination_are_not_counted_twice():
    """防止编译出的集中力节点与原杆间集中力同时重建，组合系数也必须只乘一次。"""
    session=_loaded_t()
    session.model['load_cases'][0]['member_spans'].append(dict(member=2,kind='point',w1=[0,-80,0],a=.2))
    session.model['combos']=[dict(name='C', factors={'G':1.5})]
    assert session.solve_model().ok
    base=prepare_joint_spec(session,4,case='G',loading_method='displacement')
    combined=prepare_joint_spec(session,4,case='C',loading_method='displacement')
    f,m=local_load_resultant(base.local_loads)
    fc,mc=local_load_resultant(combined.local_loads)
    length=base.arms[1].length_mm
    assert f[1] == pytest.approx(-2*length-80)
    assert m[2] == pytest.approx(-5e6-length**2-16000)
    assert fc == pytest.approx(np.asarray(f)*1.5)
    assert mc == pytest.approx(np.asarray(m)*1.5)


def _surface_source(kind='point'):
    source=dict(kind=kind,point_mm=[.25,.25,.25],radius_mm=2.,force_n=[2.,-3.,4.],moment_nmm=[5.,6.,-7.])
    if kind=='line':
        source.update(direction=[1.,0.,0.],axial_range_mm=[.1,.6])
    return source


def test_surface_loading_preserves_six_components_and_excludes_cut_nodes():
    """防止集中力矩变成单节点假峰值，且不能把局部外载直接写到驱动切面。"""
    nodes=_unit_tet(); elements=np.arange(10).reshape(1,10)
    source=_surface_source()
    force,regions=regional_load_forces(nodes.tolist(),elements.tolist(),[dict(direction=[1,0,0],length_mm=1.)],[source])
    force=np.asarray(force)
    assert force.sum(axis=0) == pytest.approx(source['force_n'],abs=1e-12)
    assert np.cross(nodes-np.asarray(source['point_mm']),force).sum(axis=0) == pytest.approx(source['moment_nmm'],abs=1e-12)
    assert force[1] == pytest.approx(np.zeros(3))
    assert 1 not in regions[0]['node_indices']


def test_line_loading_uses_its_band_and_preserves_exact_resultants():
    """防止只传线荷载的合力而丢掉线性强度产生的合矩，或把它扩散到整根杆。"""
    nodes=_unit_tet();source=_surface_source('line')
    force,regions=regional_load_forces(nodes.tolist(),[list(range(10))],[],[source])
    force=np.asarray(force)
    assert force.sum(axis=0) == pytest.approx(source['force_n'],abs=1e-12)
    assert np.cross(nodes-np.asarray(source['point_mm']),force).sum(axis=0) == pytest.approx(source['moment_nmm'],abs=1e-12)
    assert all(.1 <= nodes[i,0] <= .6 for i in regions[0]['node_indices'])


def test_loads_outside_the_retained_domain_are_not_added_again():
    """防止原整体模型梁端荷载在切面运动之外又被当作局部外载重复施加。"""
    session=_cantilever()
    spec=prepare_joint_spec(session,2,loading_method='displacement')
    assert collect_local_loads(session,2,spec.arms,'LC') == ()


def test_abaqus_script_uses_the_same_surface_algorithm_and_local_cloads():
    """防止自研支持加载节点后，Abaqus仍漏载或导出Python2不支持的语法。"""
    spec=prepare_joint_spec(_loaded_t(),4,loading_method='displacement',geometry_mode='transition')
    script=_script_text(spec,[24,16,8],'out.json')
    script.encode('ascii');compile(script,'joint_loads.py','exec')
    assert "lines = ['*CLOAD']" in script
    assert "regional_load_forces(coordinates, elements, SPEC['arms'], SPEC['local_loads'])" in script
    assert "local_load_audit_%d.json" in script
    def external_sum(*args,**kwargs):
        raise TypeError('CAE wildcard sum does not accept generators')
    namespace={'sum':external_sum};exec(abaqus_load_helper(),namespace)
    nodes=_unit_tet().tolist();sources=[_surface_source()]
    assert namespace['regional_load_forces'](nodes,[list(range(10))],[],sources)[0] == regional_load_forces(nodes,[list(range(10))],[],sources)[0]


def test_quadratic_display_surface_keeps_original_cell_ids_and_input_arrays():
    """防止增强曲面显示破坏点选单元映射，或把显示平滑写回求解网格。"""
    pv=pytest.importorskip('pyvista')
    from viz_theme import solid_surface
    points=_unit_tet();points[4,2]=.1
    grid=pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA:np.arange(10).reshape(1,10)},points)
    grid.cell_data['vtkOriginalCellIds']=[99]
    before=grid.points.copy()
    surface=solid_surface(grid)
    assert surface.n_cells>4 and surface.n_points>=10
    assert np.unique(surface.cell_data['vtkOriginalCellIds']) == [0]
    assert grid.cell_data['vtkOriginalCellIds'] == [99]
    assert grid.points == pytest.approx(before)
    assert np.any(np.linalg.norm(surface.points-points[4],axis=1)<1e-7)


def test_legacy_abaqus_console_decoding_preserves_the_real_failure(tmp_path,monkeypatch):
    """防止旧版CAE的GBK控制台被UTF8解码线程丢弃，导致用户看不到真实建模错误。"""
    import abaqus_backend
    import solid_joint
    from types import SimpleNamespace
    monkeypatch.setattr(abaqus_backend,'find_abaqus',lambda:'abaqus.bat')
    monkeypatch.setattr(abaqus_backend,'is_amd_cpu',lambda *args,**kwargs:False)
    def failed(argv,**options):
        assert options['errors']=='replace'
        return SimpleNamespace(returncode=1,stdout='',stderr='Legacy CAE geometry error')
    monkeypatch.setattr(solid_joint.subprocess,'run',failed)
    with pytest.raises(solid_joint.SolidJointError,match='Legacy CAE geometry error'):
        solid_joint.run_joint_analysis(_l_joint(),2,mesh_sizes_mm=[12,10,8],
                                      geometry_mode='intersection',output_dir=tmp_path)


def test_real_loaded_node_balances_and_retains_default_displacement_mode(tmp_path,monkeypatch):
    """真实加载节点必须完成求解、重建外载且平衡，不能靠切换force绕过报错。"""
    pytest.importorskip('gmsh')
    import native_joint
    session=_l_joint(section=sections.rectangle('R',.04,.06),load_at_joint=True)
    assert session.set_nodal_load(2,[0,0,-100,0,10,0],case_name='LC1').ok
    assert session.solve_model().ok
    monkeypatch.setattr(native_joint,'_write_vtu_and_png',lambda *args:(None,None))
    model=copy.deepcopy(session.model)
    result=native_joint.run_native_joint_analysis(session,2,mesh_sizes_mm=[30],output_dir=tmp_path)
    row=result['meshes'][0]
    assert result['loading_method']=='displacement'
    assert row['local_applied_force_n'] == pytest.approx([0,0,-100],abs=1e-7)
    assert row['local_applied_moment_nmm'] == pytest.approx([0,10000,0],abs=1e-4)
    assert np.linalg.norm(row['force_balance_n'])<1e-5
    assert np.linalg.norm(row['moment_balance_nmm'])<1e-2
    assert row['cut_motion_error_mm']<1e-8
    assert session.model==model
